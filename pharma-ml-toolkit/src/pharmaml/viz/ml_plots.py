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
    "plot_edf",
    "plot_feature_importance",
    "plot_gp_uncertainty",
    "plot_intermediate_values",
    "plot_learning_curve",
    "plot_model_comparison",
    "plot_optimization_history",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_permutation_heatmap",
    "plot_permutation_importance",
    "plot_pr_curves",
    "plot_predicted_vs_actual",
    "plot_prediction_intervals",
    "plot_residuals",
    "plot_roc_curves",
    "plot_shap_beeswarm",
    "plot_shap_dependence",
    "plot_shap_force",
    "plot_shap_group_difference",
    "plot_shap_heatmap",
    "plot_shap_importance",
    "plot_shap_waterfall",
    "plot_signed_correlations",
    "plot_slice",
    "plot_timeline",
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


def plot_shap_importance(
    summary: Any,
    top_n: int | None = 20,
    title: str = "SHAP importance (mean |value|)",
) -> Any:
    """Top-N mean-|SHAP| horizontal bars -- ``D2.py``'s own top-20 bar.

    The beeswarm's compact sibling: one bar per feature, the mean
    magnitude of the feature's SHAP values (how hard the feature
    pushes the model output, either direction), largest first.  Accepts
    the :class:`~pharmaml.ml.explain.ShapSummary` record (or its
    ``as_frame`` output with ``feature`` / ``mean_abs_shap`` columns)
    -- the plotting layer never needs shap installed.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record, or a
        DataFrame with ``feature`` / ``mean_abs_shap`` columns.
    top_n:
        How many features to show (largest mean |SHAP| first);
        ``None`` shows every feature.  Validated by
        :func:`_check_top_n`.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` carries neither the record attributes nor the
        DataFrame columns, or ``top_n`` is invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> record = ShapSummary(
    ...     names=["a", "b"], mean_abs=np.array([1.5, 0.5]),
    ...     values=np.zeros((4, 2)), feature_values=np.zeros((4, 2)),
    ...     base_value=None, model_name="GBR", task="regression", n_samples=4)
    >>> fig = plot_shap_importance(record)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]  # bottom -> top
    ['b', 'a']
    >>> ax.get_xlabel()
    'mean |SHAP value|'
    """
    _check_top_n(top_n)
    if hasattr(summary, "names") and hasattr(summary, "mean_abs"):
        names = [str(name) for name in summary.names]
        means = np.asarray(summary.mean_abs, dtype=float)
    elif isinstance(summary, pd.DataFrame) and {
        "feature",
        "mean_abs_shap",
    }.issubset(summary.columns):
        names = [str(name) for name in summary["feature"].tolist()]
        means = summary["mean_abs_shap"].to_numpy(dtype=float)
    else:
        raise DataValidationError(
            "summary must be a ShapSummary record (names/mean_abs) or a "
            "DataFrame with feature/mean_abs_shap columns, got "
            f"{type(summary).__name__}"
        )
    if len(names) != len(means):
        raise DataValidationError(
            f"names carries {len(names)} label(s) for {len(means)} mean(s)"
        )
    keep = len(means) if top_n is None else min(int(top_n), len(means))
    # largest mean first (stable, first-declared-wins ties), reversed
    # for display so the strongest feature sits at the top.
    order = np.argsort(-means, kind="stable")[:keep][::-1]

    fig, ax = new_figure(figsize=(8.4, 0.42 * len(order) + 1.7))
    positions = np.arange(len(order))
    ax.barh(
        positions,
        means[order],
        color=PALETTE[0],
        alpha=0.88,
        height=0.62,
        edgecolor="white",
    )
    for pos, value in zip(positions, means[order]):
        ax.text(value, pos, f" {value:.4g}", va="center", ha="left", fontsize=9)
    ax.axvline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_yticks(positions, [names[int(index)] for index in order])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    return fig


def plot_shap_beeswarm(
    summary: Any,
    top_n: int | None = 20,
    *,
    jitter: float = 0.16,
    seed: int = 42,
    point_size: float = 14.0,
    title: str = "SHAP beeswarm",
) -> Any:
    """The SHAP beeswarm in PURE matplotlib -- ``D2.py``'s GAP-5 figure.

    One row per feature (largest mean |SHAP| at the top), one jittered
    point per (sample, feature): the x position is the sample's SHAP
    value (how far the feature pushed THIS prediction from the base),
    the colour is the feature's own value on that sample (the house
    amber->teal ramp -- teal = high values, amber = low), exactly the
    information the original's ``shap.plots.beeswarm`` carried, drawn
    without a single shap plotting import so the viz layer stays
    dependency-free.  The jitter is seeded (``seed``, default 42) so
    two runs of the same summary are byte-identical figures.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record (needs the
        full ``values`` matrix and ``feature_values`` -- the DataFrame
        form is refused with the typed error; the bar chart is its
        sibling).
    top_n:
        How many feature rows to show; ``None`` shows every feature.
    jitter:
        Vertical scatter width per row (0 collapses to a line).
    seed:
        Seed for the row jitter (the house determinism rule).
    point_size:
        Marker size.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a ShapSummary record, its arrays
        disagree in shape, or ``top_n``/``jitter`` are invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> values = np.array([[1.0, -0.5], [0.5, 0.25], [-1.0, 0.5]])
    >>> record = ShapSummary(
    ...     names=["a", "b"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=values,
    ...     base_value=0.0, model_name="GBR", task="regression", n_samples=3)
    >>> fig = plot_shap_beeswarm(record)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]  # bottom -> top
    ['b', 'a']
    >>> ax.get_xlabel()  # the base value rides the caption when present
    'SHAP value (impact on model output; base = +0)'
    """
    _check_top_n(top_n)
    if not (hasattr(summary, "values") and hasattr(summary, "feature_values")):
        raise DataValidationError(
            "plot_shap_beeswarm needs the ShapSummary RECORD (the full "
            "values matrix + feature_values) -- the DataFrame form carries "
            "only the bar chart's means; use plot_shap_importance for that"
        )
    names = [str(name) for name in summary.names]
    values = np.asarray(summary.values, dtype=float)
    feature_values = np.asarray(summary.feature_values, dtype=float)
    if values.ndim != 2 or values.shape != feature_values.shape:
        raise DataValidationError(
            f"values shape {values.shape} must match feature_values shape "
            f"{feature_values.shape} (n_samples x n_features)"
        )
    if len(names) != values.shape[1]:
        raise DataValidationError(
            f"names carries {len(names)} label(s) for {values.shape[1]} column(s)"
        )
    if not 0.0 <= jitter <= 1.0:
        raise DataValidationError(
            f"jitter must be within [0, 1], got {jitter}"
        )
    mean_abs = np.abs(values).mean(axis=0)
    keep = len(names) if top_n is None else min(int(top_n), len(names))
    # largest mean |SHAP| first, reversed for display (top of the plot).
    order = np.argsort(-mean_abs, kind="stable")[:keep][::-1]
    n_rows = len(order)

    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = new_figure(figsize=(8.6, 0.55 * n_rows + 1.9))
    rng = np.random.default_rng(seed)
    cmap = LinearSegmentedColormap.from_list(
        "pharmaml_feature_value", [PALETTE[1], "#FFFFFF", PALETTE[0]]
    )
    scatter_handles = None
    for row, feature_index in enumerate(order):
        column = values[:, int(feature_index)]
        raw = feature_values[:, int(feature_index)]
        # per-feature value normalisation for the colour ramp: the
        # min/max of THIS feature (a feature-spanning normalisation
        # would paint every row the same half-scale grey).
        span = float(np.max(raw) - np.min(raw))
        normalized = np.zeros_like(raw) if span == 0.0 else (raw - np.min(raw)) / span
        y = np.full_like(column, float(row)) + rng.uniform(
            -jitter, jitter, size=column.shape
        )
        handle = ax.scatter(
            column,
            y,
            c=normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            alpha=0.75,
            linewidths=0.0,
        )
        if scatter_handles is None:
            scatter_handles = handle
    ax.axvline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_yticks(np.arange(n_rows), [names[int(index)] for index in order])
    ax.set_ylim(-0.75, n_rows - 0.25)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(title)
    if summary.base_value is not None:
        ax.set_xlabel(
            f"SHAP value (impact on model output; base = {summary.base_value:+.4g})"
        )
    if scatter_handles is not None:
        colorbar = fig.colorbar(scatter_handles, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("feature value (row-normalised)")
    return fig


# ---------------------------------------------------------------------------
# The round-43 SHAP gallery (100x-160): the five remaining signature
# figures of the shap repo's plot gallery, drawn in the same
# pure-matplotlib dialect as the two above -- the viz layer never
# imports shap, only the computation (ml/explain.py) does.
# ---------------------------------------------------------------------------


def _unwrap_shap_record(summary: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The ShapSummary-record contract shared by the SHAP figures.

    Returns ``(names, values, feature_values)`` after the same shape
    checks :func:`plot_shap_beeswarm` runs; the ``as_frame`` DataFrame
    form is refused with the pointer to the bar chart (its means-only
    sibling), the message the beeswarm established.
    """
    if not (
        hasattr(summary, "names")
        and hasattr(summary, "values")
        and hasattr(summary, "feature_values")
    ):
        raise DataValidationError(
            "this figure needs the ShapSummary RECORD (the full values "
            "matrix + feature_values) -- the DataFrame form carries only "
            "the bar chart's means; use plot_shap_importance for that"
        )
    names = [str(name) for name in summary.names]
    values = np.asarray(summary.values, dtype=float)
    feature_values = np.asarray(summary.feature_values, dtype=float)
    if values.ndim != 2 or values.shape != feature_values.shape:
        raise DataValidationError(
            f"values shape {values.shape} must match feature_values shape "
            f"{feature_values.shape} (n_samples x n_features)"
        )
    if len(names) != values.shape[1]:
        raise DataValidationError(
            f"names carries {len(names)} label(s) for {values.shape[1]} column(s)"
        )
    return names, values, feature_values


def _sample_index(sample: Any, n_samples: int) -> int:
    """Typed guard for the local-explanation row index (waterfall/force)."""
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)):
        raise DataValidationError(
            f"sample must be an integer row index, got {sample!r}"
        )
    index = int(sample)
    if not 0 <= index < n_samples:
        raise DataValidationError(
            f"sample index {index} is out of range for {n_samples} row(s) -- "
            f"valid range 0..{n_samples - 1}"
        )
    return index


def plot_shap_waterfall(
    summary: Any,
    sample: int = 0,
    *,
    max_display: int = 10,
    title: str = "SHAP waterfall",
) -> Any:
    """The LOCAL explanation: the base-value -> prediction walk (100x-160).

    ``shap.plots.waterfall``'s signature figure in the house dialect:
    ONE row of the summary, its SHAP contributions drawn as a
    cumulative horizontal-bar walk from the explainer's expected
    output (``summary.base_value``) to the row's prediction
    (``base + sum(values)``) -- POSITIVE bars in the palette's teal
    (``PALETTE[0]``), NEGATIVE ones in its amber (``PALETTE[1]``),
    largest |value| first, thin dashed connectors tying each bar's end
    to the next bar's start, and dashed vertical lines marking both
    ends.  Features beyond ``max_display`` collapse into one summed
    "Σ remaining N features" bar, exactly the upstream collapse.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record WITH a
        ``base_value`` (the walk's starting point) -- a summary that
        carries none is refused with the typed error.
    sample:
        The row to explain (0-based index into the values matrix).
    max_display:
        How many feature bars to draw before the summed remainder;
        validated by :func:`_check_top_n`.
    title:
        Chart title (the explained row index rides it).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a ShapSummary record, carries no
        ``base_value``, ``sample`` is out of range, or ``max_display``
        is invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> values = np.array([[1.25, -0.5, 0.25], [0.1, 0.2, -0.1]])
    >>> record = ShapSummary(
    ...     names=["a", "b", "c"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ...     base_value=0.25, model_name="GBR", task="regression", n_samples=2)
    >>> fig = plot_shap_waterfall(record)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]  # bottom -> top
    ['c = 3.00', 'b = 2.00', 'a = 1.00']
    >>> ends = {round(float(line.get_xdata()[0]), 6) for line in ax.lines
    ...         if len(line.get_xdata()) == 2}
    >>> 0.25 in ends and 1.25 in ends   # the base and the prediction f(x)
    True
    """
    _check_top_n(max_display)
    names, values, feature_values = _unwrap_shap_record(summary)
    base_value = getattr(summary, "base_value", None)
    if base_value is None:
        raise DataValidationError(
            "plot_shap_waterfall needs summary.base_value -- the waterfall "
            "walks from the explainer's expected output; this summary "
            "carries none"
        )
    base = float(base_value)
    index = _sample_index(sample, values.shape[0])
    row = values[index]
    data_row = feature_values[index]

    order = np.argsort(-np.abs(row), kind="stable")
    keep = min(int(max_display), len(row))
    # y tick labels carry the sample's own feature value (shap's
    # show_data convention): "a = 1.24".
    labels = [f"{names[int(i)]} = {data_row[int(i)]:.2f}" for i in order[:keep]]
    amounts = [float(row[int(i)]) for i in order[:keep]]
    remainder_count = len(row) - keep
    if remainder_count > 0:
        labels.append(
            f"Σ remaining {remainder_count} feature"
            f"{'' if remainder_count == 1 else 's'}"
        )
        amounts.append(float(row[order[keep:]].sum()))

    fig, ax = new_figure(figsize=(8.6, 0.55 * len(labels) + 2.0))
    # largest |value| first in the item list, so its bar gets the TOP
    # row (positions reversed for display -- the house convention).
    positions = np.arange(len(labels))[::-1]
    cumulative = base
    ends: list[tuple[float, float]] = []
    for position, amount in zip(positions, amounts):
        color = PALETTE[0] if amount >= 0 else PALETTE[1]
        left = cumulative if amount >= 0 else cumulative + amount
        ax.barh(
            position,
            abs(amount),
            left=left,
            height=0.62,
            color=color,
            alpha=0.9,
            edgecolor="white",
        )
        ax.text(
            left + abs(amount) if amount >= 0 else left,
            position,
            f" {amount:+.2f}",
            va="center",
            ha="left" if amount >= 0 else "right",
            fontsize=9,
        )
        ends.append((cumulative + amount, float(position)))
        cumulative += amount
    # thin dashed connectors between consecutive bar ends
    for step in range(len(ends) - 1):
        x_end, y_cur = ends[step]
        y_next = ends[step + 1][1]
        ax.plot(
            [x_end, x_end], [y_next + 0.31, y_cur - 0.31],
            ls="--", lw=0.7, color="#9CA3AF", zorder=1,
        )
    prediction = base + float(row.sum())
    ax.axvline(base, color="#525252", lw=1.0, ls="--", zorder=2)
    ax.axvline(prediction, color="#525252", lw=1.0, ls="--", zorder=2)
    ax.set_yticks(np.arange(len(labels)), labels[::-1])
    lo = min(base, min(x for x, _ in ends))
    hi = max(base, prediction, max(x for x, _ in ends))
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.07 * span, hi + 0.07 * span)
    ax.set_xlabel("SHAP contribution (model output units)")
    ax.set_title(f"{title} — sample {index} of {values.shape[0]}")
    return fig


def _equal_width_bins(column: np.ndarray, bins: int) -> np.ndarray:
    """Equal-width bin indices, clamped into ``[0, bins - 1]``."""
    lo = float(np.min(column))
    hi = float(np.max(column))
    if hi <= lo:
        return np.zeros(len(column), dtype=int)
    indices = ((column - lo) / (hi - lo) * bins).astype(int)
    return np.clip(indices, 0, bins - 1)


def _binned_r2(x: np.ndarray, color: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """``R^2`` of the 2-D binned cell-mean prediction of ``y``.

    Both columns discretised into ``bins`` equal-width bins, the cell
    means of ``y`` looked up per observation, and the residual sum of
    squares scored against the total -- the shared engine of the
    auto-interaction pick below.
    """
    x_bins = _equal_width_bins(x, bins)
    c_bins = _equal_width_bins(color, bins)
    sums = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))
    np.add.at(sums, (x_bins, c_bins), y)
    np.add.at(counts, (x_bins, c_bins), 1.0)
    predicted = np.where(counts > 0, sums / np.maximum(counts, 1.0), y.mean())
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0.0:
        return 0.0
    sse_2d = float(np.sum((y - predicted[x_bins, c_bins]) ** 2))
    return 1.0 - sse_2d / total


def _auto_interaction(
    names: list[str],
    feature_index: int,
    values: np.ndarray,
    feature_values: np.ndarray,
) -> str | None:
    """Pick the strongest interacting feature, or ``None``.

    shap's ``approximate_interactions`` heuristic (its ``utils/legacy``
    module), honestly documented as such: for every candidate
    ``k != feature`` BOTH the x column and the candidate are
    discretised into 10 equal-width bins, the 2-D cell-mean prediction
    of y scores ``R^2 = 1 - SSE_2d / SSE_total``, and the best
    candidate wins (ties resolve first-declared).  A CONSTANT
    candidate column scores 0 (no spread, no explanatory power); when
    every candidate is degenerate the answer is ``None`` -- no honest
    colour exists.
    """
    x = feature_values[:, feature_index]
    y = values[:, feature_index]
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0.0:
        return None
    best_name: str | None = None
    best_score = 0.0
    for candidate in range(len(names)):
        if candidate == feature_index:
            continue
        column = feature_values[:, candidate]
        if float(np.max(column) - np.min(column)) == 0.0:
            continue  # a constant candidate scores 0 by definition
        score = _binned_r2(x, column, y)
        if score > best_score:  # strict: first-declared wins ties
            best_name, best_score = names[candidate], score
    return best_name


def plot_shap_dependence(
    summary: Any,
    feature: str,
    *,
    color_by: str | None = None,
    jitter: float = 0.0,
    point_size: float = 22.0,
    title: str = "SHAP dependence",
) -> Any:
    """One feature's value vs its SHAP value, coloured by its partner (100x-160).

    ``shap.plots.scatter``'s dependence figure: the x axis is the
    feature's own value, the y axis its SHAP value, and the COLOUR is
    the interacting feature's value on the same row (the house
    amber->teal ramp + colorbar, the beeswarm's construction).  The
    colour source is either explicit (``color_by="b"``) or AUTO-PICKED
    by shap's ``approximate_interactions`` heuristic when left as
    ``None`` -- the strongest binned 2-D ``R^2`` candidate (see
    :func:`_auto_interaction`); ``color_by="none"`` (or an all-degenerate
    candidate set) falls back to a plain single-colour scatter.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record (the full
        values matrix + feature_values).
    feature:
        The feature to plot; must be one of ``summary.names``.
    color_by:
        The interacting feature to colour by; ``None`` auto-picks,
        ``"none"`` refuses colour.  Must differ from ``feature`` and
        exist in ``summary.names``.
    jitter:
        Seeded uniform x-jitter as a fraction of the x range (the
        beeswarm's determinism rule, ``default_rng(42)``); ``0``
        plots the raw values.
    point_size:
        Marker size.
    title:
        Chart title (the honest "no interacting colour found" note
        rides it when the auto-pick finds nothing).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a ShapSummary record, ``feature`` /
        ``color_by`` name an unknown feature or the same feature, or
        ``jitter`` sits outside ``[0, 1]``.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> a = np.arange(8.0)
    >>> b = np.tile([0.0, 1.0], 4)
    >>> shap_a = b.copy()               # a's SHAP value tracks b: the interaction
    >>> values = np.column_stack([shap_a, np.zeros(8)])
    >>> record = ShapSummary(
    ...     names=["a", "b"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=np.column_stack([a, b]),
    ...     base_value=None, model_name="GBR", task="regression", n_samples=8)
    >>> fig = plot_shap_dependence(record, "a")
    >>> ax = fig.axes[0]
    >>> (ax.get_xlabel(), ax.get_ylabel())
    ('a', 'SHAP value')
    >>> fig.axes[1].get_ylabel()        # the colorbar names the auto-picked colour
    'b value'
    """
    names, values, feature_values = _unwrap_shap_record(summary)
    if feature not in names:
        raise DataValidationError(
            f"feature {feature!r} is not one of the summary's features {names}"
        )
    color_name: str | None = None
    if isinstance(color_by, str) and color_by != "none":
        if color_by == feature:
            raise DataValidationError(
                f"color_by {color_by!r} must differ from feature {feature!r} "
                "-- a feature cannot colour its own dependence plot"
            )
        if color_by not in names:
            raise DataValidationError(
                f"color_by {color_by!r} is not one of the summary's "
                f"features {names}"
            )
        color_name = color_by
    if not 0.0 <= jitter <= 1.0:
        raise DataValidationError(f"jitter must be within [0, 1], got {jitter}")

    feature_index = names.index(feature)
    x = feature_values[:, feature_index]
    y = values[:, feature_index]
    auto_attempted = color_by is None
    if color_name is None and auto_attempted:
        color_name = _auto_interaction(names, feature_index, values, feature_values)

    rng = np.random.default_rng(42)
    x_span = float(np.max(x) - np.min(x))
    x_plot = x + rng.uniform(-jitter, jitter, size=x.shape) * (
        x_span if x_span > 0.0 else 1.0
    )

    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = new_figure(figsize=(7.8, 5.6))
    if color_name is not None:
        color_index = names.index(color_name)
        color_column = feature_values[:, color_index]
        c_span = float(np.max(color_column) - np.min(color_column))
        normalized = (
            np.zeros_like(color_column)
            if c_span == 0.0
            else (color_column - np.min(color_column)) / c_span
        )
        cmap = LinearSegmentedColormap.from_list(
            "pharmaml_feature_value", [PALETTE[1], "#FFFFFF", PALETTE[0]]
        )
        handle = ax.scatter(
            x_plot,
            y,
            c=normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            alpha=0.8,
            linewidths=0.0,
        )
        colorbar = fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(f"{color_name} value")
    else:
        ax.scatter(
            x_plot, y, color=PALETTE[0], s=point_size, alpha=0.8, linewidths=0.0
        )
        if auto_attempted:
            title = f"{title} (no interacting colour found)"
    ax.axhline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(title)
    return fig


def plot_shap_heatmap(
    summary: Any,
    *,
    max_display: int = 10,
    title: str = "SHAP heatmap",
) -> Any:
    """The instances x features SHAP matrix, clustered (100x-160).

    ``shap.plots.heatmap``'s figure: rows are INSTANCES ordered by
    hierarchical clustering (complete linkage over the |SHAP| rows,
    ``scipy.cluster.hierarchy`` -- already a core dependency), columns
    are the top ``max_display`` features by mean |SHAP| (strongest
    first), cells carry the SIGNED SHAP values on the module's
    diverging red->white->teal ramp (zero-centred, the
    ``plot_permutation_heatmap`` construction), and a thin top bar
    panel shows the mean |SHAP| of every displayed feature -- the
    house multi-panel pattern with ``height_ratios`` ``[1, 7]``.

    The honest scope: with fewer than two instances the clustering
    cannot run (linkage needs >= 2 rows) and the input order is kept.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record (the full
        values matrix).
    max_display:
        How many feature columns to show; validated by
        :func:`_check_top_n`.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a ShapSummary record or ``max_display``
        is invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> values = np.array([
    ...     [ 2.0,  0.1, -0.1],
    ...     [ 1.8,  0.0,  0.0],
    ...     [-2.0, -0.1,  0.1],
    ...     [-1.8,  0.0,  0.0]])
    >>> record = ShapSummary(
    ...     names=["a", "b", "c"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=values,
    ...     base_value=None, model_name="GBR", task="regression", n_samples=4)
    >>> fig = plot_shap_heatmap(record)
    >>> heat = fig.axes[1]            # axes[0] is the importance bar
    >>> [tick.get_text() for tick in heat.get_xticklabels()]  # strongest first
    ['a', 'b', 'c']
    >>> len(fig.axes) >= 2            # the importance bar + the heatmap (+ cbar)
    True
    """
    _check_top_n(max_display)
    names, values, _feature_values = _unwrap_shap_record(summary)
    mean_abs = np.abs(values).mean(axis=0)
    keep = min(int(max_display), len(names))
    feature_order = np.argsort(-mean_abs, kind="stable")[:keep]

    n_samples = int(values.shape[0])
    abs_rows = np.abs(values[:, feature_order])
    instance_order = np.arange(n_samples)
    if n_samples >= 2:
        from scipy.cluster.hierarchy import leaves_list, linkage

        try:
            instance_order = np.asarray(
                leaves_list(linkage(abs_rows, method="complete")), dtype=int
            )
        except ValueError:  # degenerate distances -> keep the input order
            instance_order = np.arange(n_samples)
    matrix = values[np.ix_(instance_order, feature_order)]

    from matplotlib.colors import LinearSegmentedColormap

    fig, (ax_bar, ax_heat) = new_figure(
        2,
        1,
        figsize=(max(6.4, 2.2 + 1.05 * keep), max(3.6, 1.1 + 0.34 * n_samples)),
        gridspec_kw={"height_ratios": [1, 7], "hspace": 0.08},
    )
    ax_bar.bar(
        np.arange(keep),
        mean_abs[feature_order],
        color=PALETTE[0],
        alpha=0.88,
        width=0.62,
        edgecolor="white",
    )
    ax_bar.set_xlim(-0.5, keep - 0.5)
    ax_bar.set_ylabel("mean |SHAP|", fontsize=9)
    ax_bar.set_xticks(np.arange(keep))
    ax_bar.tick_params(labelbottom=False)
    ax_bar.set_title(title)

    cmap = LinearSegmentedColormap.from_list(
        "pharmaml_signed", [PALETTE[3], "#FFFFFF", PALETTE[0]]
    )
    bound = max(float(np.max(np.abs(matrix))), 1e-12)
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04, label="SHAP value")
    ax_heat.set_xticks(np.arange(keep), [names[int(i)] for i in feature_order],
                       fontsize=9)
    ax_heat.set_yticks(
        np.arange(n_samples), [str(int(i)) for i in instance_order], fontsize=8
    )
    ax_heat.set_xlabel("feature (strongest first)")
    ax_heat.set_ylabel("instance (clustered)")
    ax_heat.grid(False)
    return fig


def plot_shap_force(
    summary: Any,
    sample: int = 0,
    *,
    max_display: int = 10,
    title: str = "SHAP force",
) -> Any:
    """The additive force plot, static matplotlib twin (100x-160).

    ``shap.plots.force``'s LOCAL explanation on ONE row: the signed
    contributions as stacked segments -- POSITIVES stacked rightward
    from the base value (teal, ``PALETTE[0]``), NEGATIVES leftward
    (amber, ``PALETTE[1]``), largest |value| first, the tail beyond
    ``max_display`` summed into one "Σ remaining" segment.  The base
    value and the prediction ``f(x) = base + sum(values)`` are
    annotated at their positions; a horizontal centre line carries the
    row.  The JS interactive layout (hover, re-stacking, the
    additive-force animation) is honestly out of scope for a desktop
    app -- this is the static twin.

    Parameters
    ----------
    summary:
        A :class:`~pharmaml.ml.explain.ShapSummary` record WITH a
        ``base_value`` -- a summary that carries none is refused with
        the typed error.
    sample:
        The row to explain (0-based index into the values matrix).
    max_display:
        How many feature segments to draw before the summed remainder;
        validated by :func:`_check_top_n`.
    title:
        Chart title (the explained row index rides it).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``summary`` is not a ShapSummary record, carries no
        ``base_value``, ``sample`` is out of range, or ``max_display``
        is invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import ShapSummary
    >>> import numpy as np
    >>> values = np.array([[1.25, 0.75, 0.25]])
    >>> record = ShapSummary(
    ...     names=["a", "b", "c"], mean_abs=np.abs(values).mean(axis=0),
    ...     values=values, feature_values=np.array([[1.0, 2.0, 3.0]]),
    ...     base_value=0.5, model_name="GBR", task="regression", n_samples=1)
    >>> fig = plot_shap_force(record)
    >>> ax = fig.axes[0]
    >>> [annotation.get_text() for annotation in ax.texts]
    ['base = +0.5', 'f(x) = +2.75']
    >>> ax.get_xlabel()
    'model output'
    """
    _check_top_n(max_display)
    _names, values, _feature_values = _unwrap_shap_record(summary)
    base_value = getattr(summary, "base_value", None)
    if base_value is None:
        raise DataValidationError(
            "plot_shap_force needs summary.base_value -- the force plot "
            "stacks from the explainer's expected output; this summary "
            "carries none"
        )
    base = float(base_value)
    index = _sample_index(sample, values.shape[0])
    row = values[index]

    order = np.argsort(-np.abs(row), kind="stable")
    keep = min(int(max_display), len(row))
    amounts = [float(row[int(i)]) for i in order[:keep]]
    remainder_count = len(row) - keep
    if remainder_count > 0:
        amounts.append(float(row[order[keep:]].sum()))
    prediction = base + float(row.sum())

    fig, ax = new_figure(figsize=(9.2, 2.6))
    positive_edge = base
    negative_edge = prediction
    for amount in amounts:
        if amount >= 0:
            ax.barh(
                0.0, amount, left=positive_edge, height=0.5,
                color=PALETTE[0], alpha=0.9, edgecolor="white",
            )
            positive_edge += amount
        else:
            ax.barh(
                0.0, -amount, left=negative_edge + amount, height=0.5,
                color=PALETTE[1], alpha=0.9, edgecolor="white",
            )
            negative_edge += amount
    ax.axhline(0.0, color="#525252", lw=0.9)
    ax.axvline(prediction, color="#525252", lw=0.9, ls="--", zorder=2)
    ax.annotate(
        f"base = {base:+.4g}",
        xy=(base, 0.27), xytext=(base, 0.62),
        ha="center", va="center", fontsize=9,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#9CA3AF"},
    )
    ax.annotate(
        f"f(x) = {prediction:+.4g}",
        xy=(prediction, 0.27), xytext=(prediction, 0.62),
        ha="center", va="center", fontsize=9,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#9CA3AF"},
    )
    ax.set_yticks([])
    lo = min(base, negative_edge)
    hi = max(base, positive_edge, prediction)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.08 * span, hi + 0.08 * span)
    ax.set_ylim(-0.8, 0.95)
    ax.set_xlabel("model output")
    ax.set_title(f"{title} — sample {index} of {values.shape[0]}")
    return fig


def plot_shap_group_difference(
    difference: Any,
    *,
    top_n: int = 10,
    title: str = "SHAP group difference",
) -> Any:
    """Cohort mean-SHAP difference bars with bootstrap whiskers (100x-160).

    ``shap.plots.group_difference``'s figure: per feature, the cohort's
    mean SHAP minus the rest's -- horizontal bars sorted by
    |difference| (largest at the top), POSITIVE in teal / NEGATIVE in
    amber, whiskers spanning the record's bootstrap ``low`` to
    ``high``, a vertical zero line, and the cohort label with its
    ``n_group``/``n_rest`` counts riding the title.  Accepts the
    :class:`~pharmaml.ml.explain.GroupDifference` record OR its
    ``as_frame`` DataFrame (``feature`` / ``difference`` / ``low`` /
    ``high`` columns) -- the same dual-form contract as
    :func:`plot_shap_importance`.

    Parameters
    ----------
    difference:
        A :class:`~pharmaml.ml.explain.GroupDifference` record, or a
        DataFrame with ``feature`` / ``difference`` / ``low`` /
        ``high`` columns.
    top_n:
        How many features to show (largest |difference| first);
        validated by :func:`_check_top_n`.
    title:
        Chart title (the label + cohort counts ride it).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``difference`` carries neither the record attributes nor
        the DataFrame columns, or ``top_n`` is invalid.

    Examples
    --------
    >>> from pharmaml.ml.explain import GroupDifference
    >>> import numpy as np
    >>> record = GroupDifference(
    ...     names=["a", "b"], differences=np.array([0.8, -0.3]),
    ...     low=np.array([0.5, -0.6]), high=np.array([1.1, 0.0]),
    ...     n_group=6, n_rest=4, label="cohort")
    >>> fig = plot_shap_group_difference(record)
    >>> ax = fig.axes[0]
    >>> [tick.get_text() for tick in ax.get_yticklabels()]  # bottom -> top
    ['b', 'a']
    >>> ax.get_xlabel()
    'mean SHAP difference (group - rest)'
    """
    _check_top_n(top_n)
    label: str | None = None
    n_group = 0
    n_rest = 0
    if hasattr(difference, "names") and hasattr(difference, "differences"):
        names = [str(name) for name in difference.names]
        diffs = np.asarray(difference.differences, dtype=float)
        low = np.asarray(difference.low, dtype=float)
        high = np.asarray(difference.high, dtype=float)
        label = str(getattr(difference, "label", "group"))
        n_group = int(getattr(difference, "n_group", 0))
        n_rest = int(getattr(difference, "n_rest", 0))
    elif isinstance(difference, pd.DataFrame) and {
        "feature",
        "difference",
        "low",
        "high",
    }.issubset(difference.columns):
        names = [str(name) for name in difference["feature"].tolist()]
        diffs = difference["difference"].to_numpy(dtype=float)
        low = difference["low"].to_numpy(dtype=float)
        high = difference["high"].to_numpy(dtype=float)
    else:
        raise DataValidationError(
            "difference must be a GroupDifference record (names/differences/"
            "low/high) or a DataFrame with feature/difference/low/high "
            f"columns, got {type(difference).__name__}"
        )
    if not (len(names) == len(diffs) == len(low) == len(high)):
        raise DataValidationError(
            f"names carries {len(names)} label(s) for {len(diffs)} "
            f"difference(s) with {len(low)} low and {len(high)} high bound(s)"
        )
    if diffs.size == 0:
        raise DataValidationError(
            "the group difference is empty -- pass the GroupDifference "
            "record a group_difference() call returns"
        )
    keep = len(diffs) if top_n is None else min(int(top_n), len(diffs))
    order = np.argsort(-np.abs(diffs), kind="stable")[:keep][::-1]

    fig, ax = new_figure(figsize=(8.6, 0.5 * len(order) + 1.9))
    positions = np.arange(len(order))
    colors = [PALETTE[0] if diffs[int(i)] >= 0 else PALETTE[1] for i in order]
    ax.barh(
        positions,
        diffs[order],
        color=colors,
        alpha=0.88,
        height=0.58,
        edgecolor="white",
    )
    # bootstrap whiskers from low to high, capped at the point estimate
    lower = np.maximum(diffs[order] - low[order], 0.0)
    upper = np.maximum(high[order] - diffs[order], 0.0)
    ax.errorbar(
        diffs[order],
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#374151",
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )
    ax.axvline(0.0, color="#525252", lw=0.8, ls=":")
    ax.set_yticks(positions, [names[int(i)] for i in order])
    ax.set_xlabel("mean SHAP difference (group - rest)")
    if label is not None:
        ax.set_title(f"{title} — {label} (n={n_group}) vs rest (n={n_rest})")
    else:
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


# -- unsupervised twins (PDSH 05.09-05.13, 100x-157) ------------------ #
def plot_pca_variance(
    result: Any,
    title: str = "PCA explained variance",
) -> Any:
    """Explained-variance ladder -- the "how many components?" answer.

    Bars: each component's own share; the stepped line: the running
    cumulative total; the dashed guide: the 95 % line the handbook
    recommends before trusting a projection.  The honest annotation:
    the cumulative share actually retained at the fitted k (not the
    k the user *wished* for).

    Parameters
    ----------
    result:
        A :class:`~pharmaml.ml.unsupervised.PcaResult` from
        :func:`~pharmaml.ml.unsupervised.pca_reduce`.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.ml.unsupervised import pca_reduce
    >>> rng = np.random.default_rng(42)
    >>> frame = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    >>> fig = plot_pca_variance(pca_reduce(frame))
    >>> bool(fig.axes)
    True
    """
    try:
        ratio = [float(value) for value in result.explained_variance_ratio]
        cumulative = list(np.cumsum(ratio))
        fitted = int(result.n_components)
    except (AttributeError, TypeError) as exc:
        raise DataValidationError(
            f"plot_pca_variance expects a PcaResult (from pca_reduce), got {type(result).__name__}"
        ) from exc
    if not ratio or any(value < 0 or value > 1 for value in ratio):
        raise DataValidationError("explained_variance_ratio values must lie in [0, 1]")

    positions = np.arange(1, len(ratio) + 1)
    fig, ax = new_figure(figsize=(8.4, 5.2))
    ax.bar(positions, ratio, width=0.62, color=PALETTE[0], alpha=0.85,
           edgecolor="white", linewidth=0.5, label="per component")
    ax.step(positions, cumulative, where="mid", color=PALETTE[1], lw=2.0,
            marker="s", markersize=4.5, label="cumulative")
    ax.axhline(0.95, color=PALETTE[3], linestyle="--", lw=1.4, alpha=0.7)
    ax.text(0.99, 0.965, "95 % guide", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5, color=PALETTE[3])
    verdict = (
        "enough for a trustworthy projection"
        if cumulative[-1] >= 0.95
        else f"only {cumulative[-1]:.0%} at k={fitted} -- add components or accept the loss"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"PC{i}" for i in positions], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of total variance")
    ax.set_title(f"{title}\nk={fitted} components -- {verdict}", fontsize=11)
    ax.legend(fontsize=9, loc="center right")
    return fig


def plot_kmeans_diagnostics(
    result: Any,
    title: str = "k-means diagnostics",
) -> Any:
    """Inertia elbow AND silhouette sweep -- the auditable k choice.

    Two panels: (a) the classic inertia elbow (subjective -- where
    does the bend sit?), (b) the mean silhouette per k (objective --
    closer to 1 means better-separated clusters).  The chosen k is
    marked on both, and the verdict line translates the silhouette
    into the handbook's own words.

    Parameters
    ----------
    result:
        A :class:`~pharmaml.ml.unsupervised.KMeansResult` from
        :func:`~pharmaml.ml.unsupervised.kmeans_cluster`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        chosen = int(result.k)
        scan = result.scan
        silhouette = result.silhouette
    except (AttributeError, TypeError) as exc:
        raise DataValidationError(
            f"plot_kmeans_diagnostics expects a KMeansResult (from kmeans_cluster), "
            f"got {type(result).__name__}"
        ) from exc

    if scan is None or len(scan) == 0:
        fig, ax = new_figure(figsize=(8.4, 4.6))
        message = (
            f"k={chosen} was fixed manually -- rerun with k left on 'auto' "
            "for the elbow + silhouette sweep this figure needs"
        )
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
                transform=ax.transAxes, wrap=True)
        ax.set_title(f"{title}\n(silhouette at k={chosen}: "
                     f"{'n/a' if silhouette is None else f'{silhouette:.3f}'})", fontsize=11)
        return fig

    ks = np.asarray(scan["k"], dtype=float)
    inertias = np.asarray(scan["inertia"], dtype=float)
    sils = np.asarray(scan["silhouette"], dtype=float)
    fig, (ax_a, ax_b) = new_figure(nrows=1, ncols=2, figsize=(10.8, 4.6))
    ax_a.plot(ks, inertias, marker="o", lw=1.9, color=PALETTE[0])
    ax_a.axvline(chosen, color=PALETTE[3], linestyle="--", lw=1.4, alpha=0.8)
    ax_a.set_xlabel("k")
    ax_a.set_ylabel("inertia (within-cluster SS)")
    ax_a.set_title(f"(a) elbow -- chosen k={chosen}", fontsize=10)

    mask = np.isfinite(sils)
    ax_b.plot(ks[mask], sils[mask], marker="s", lw=1.9, color=PALETTE[1])
    ax_b.axvline(chosen, color=PALETTE[3], linestyle="--", lw=1.4, alpha=0.8)
    if silhouette is not None:
        ax_b.scatter([chosen], [float(silhouette)], s=90, color=PALETTE[3],
                     edgecolor="white", linewidth=1.2, zorder=5)
    verdict = (
        "strong, well-separated structure"
        if (silhouette or 0) > 0.5
        else "reasonable structure"
        if (silhouette or 0) > 0.25
        else "weak / overlapping clusters -- k may not be the right model"
    )
    ax_b.set_xlabel("k")
    ax_b.set_ylabel("mean silhouette")
    ax_b.set_title(f"(b) silhouette -- {verdict}", fontsize=10)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_gmm_selection(
    result: Any,
    title: str = "Gaussian mixture model selection",
) -> Any:
    """AIC/BIC vs component count -- the handbook's own GMM selector.

    Both curves drop, flatten, then rise as components overfit; BIC's
    heavier complexity penalty usually bottoms at a smaller count than
    AIC.  The chosen count is marked, and the honest note names which
    criterion made the call.

    Parameters
    ----------
    result:
        A :class:`~pharmaml.ml.unsupervised.GmmResult` from
        :func:`~pharmaml.ml.unsupervised.gmm_cluster`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        chosen = int(result.n_components)
        chosen_by = str(result.chosen_by)
        scan = result.scan
    except (AttributeError, TypeError) as exc:
        raise DataValidationError(
            f"plot_gmm_selection expects a GmmResult (from gmm_cluster), "
            f"got {type(result).__name__}"
        ) from exc

    if scan is None or len(scan) == 0:
        fig, ax = new_figure(figsize=(8.4, 4.6))
        ax.text(
            0.5, 0.5,
            f"n_components={chosen} was fixed manually -- rerun with it on "
            "'auto' for the AIC/BIC sweep this figure needs",
            ha="center", va="center", fontsize=10, transform=ax.transAxes, wrap=True,
        )
        ax.set_title(title, fontsize=11)
        return fig

    counts = np.asarray(scan["n_components"], dtype=float)
    aic = np.asarray(scan["aic"], dtype=float)
    bic = np.asarray(scan["bic"], dtype=float)
    fig, ax = new_figure(figsize=(8.4, 5.2))
    ax.plot(counts, aic, marker="o", lw=1.9, color=PALETTE[0], label="AIC")
    ax.plot(counts, bic, marker="s", lw=1.9, color=PALETTE[1], label="BIC")
    ax.axvline(chosen, color=PALETTE[3], linestyle="--", lw=1.4, alpha=0.8)
    y_at = float(bic[list(counts).index(chosen)]) if chosen in list(counts) else None
    if y_at is not None:
        ax.scatter([chosen], [y_at], s=90, color=PALETTE[3],
                   edgecolor="white", linewidth=1.2, zorder=5)
    ax.set_xlabel("n_components")
    ax.set_ylabel("information criterion (lower is better)")
    verdict = (
        f"{chosen_by.upper()} prefers {chosen} component"
        f"{'s' if chosen != 1 else ''}"
        if chosen_by in {"aic", "bic"}
        else f"k={chosen} fixed manually"
    )
    ax.set_title(f"{title}\n{verdict}", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    return fig


def plot_kde_density(
    result: Any,
    values: Any = None,
    title: str = "Kernel density estimate",
) -> Any:
    """The KDE curve over its own bandwidth story.

    Filled density curve; when the raw ``values`` are supplied the
    observing histogram and rug ride underneath (the handbook's own
    hist+KDE pairing).  The annotation names the bandwidth AND how it
    was chosen -- ``cv`` (cross-validated log-likelihood), Silverman
    (small-sample fallback) or manual -- because a KDE figure without
    its bandwidth is a rumor, not a report.

    Parameters
    ----------
    result:
        A :class:`~pharmaml.ml.unsupervised.KdeResult` from
        :func:`~pharmaml.ml.unsupervised.kde_fit`.
    values:
        Optional raw observations (Series / array) for the histogram
        and rug.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.ml.unsupervised import kde_fit
    >>> rng = np.random.default_rng(42)
    >>> fig = plot_kde_density(kde_fit(pd.Series(rng.normal(size=80))))
    >>> bool(fig.axes)
    True
    """
    try:
        grid = np.asarray(result.grid, dtype=float)
        density = np.asarray(result.density, dtype=float)
        bandwidth = float(result.bandwidth)
        source = str(result.bandwidth_source)
        column = str(result.column)
    except (AttributeError, TypeError) as exc:
        raise DataValidationError(
            f"plot_kde_density expects a KdeResult (from kde_fit), got {type(result).__name__}"
        ) from exc
    if grid.size != density.size or grid.size < 2:
        raise DataValidationError("KdeResult grid and density must be equal-length (>= 2)")

    fig, ax = new_figure(figsize=(8.4, 5.0))
    if values is not None:
        raw = np.asarray(
            values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values, dtype=float
        ).ravel()
        ax.hist(raw, bins=30, density=True, color=PALETTE[2], alpha=0.28,
                edgecolor="white", linewidth=0.4, label="observed (hist)")
        ax.plot(raw, np.full(raw.shape, 0.0), "|", color=PALETTE[2], alpha=0.5,
                markersize=8, label="rug")
    ax.fill_between(grid, density, color=PALETTE[0], alpha=0.22, linewidth=0)
    ax.plot(grid, density, lw=2.2, color=PALETTE[0], label=f"KDE ({result.kernel})")
    ax.set_xlabel(column)
    ax.set_ylabel("density")
    how = {"cv": "5-fold CV log-likelihood", "silverman": "Silverman rule (n < 10)",
           "manual": "manual"}.get(source, source)
    ax.set_title(f"{title}\ncolumn {column!r} -- bandwidth {bandwidth:.3g} "
                 f"chosen by {how}", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    return fig


def plot_manifold_embedding(
    embedding: Any,
    labels: Any = None,
    title: str | None = None,
) -> Any:
    """Scatter an embedding, coloured by a target (the digits-figure pattern).

    Discrete / categorical ``labels`` colour per class with the house
    palette; a numeric target drives a colour ramp + colorbar (the
    Isomap-coloured-by-class figure the handbook's manifold chapter is
    famous for).  No labels: a single honest teal point cloud.

    Parameters
    ----------
    embedding:
        An :class:`~pharmaml.ml.manifold.EmbeddingRecord` from
        :func:`~pharmaml.ml.manifold.manifold_embedding`.
    labels:
        Optional per-sample target (Series / array), same row order.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.ml.manifold import manifold_embedding
    >>> rng = np.random.default_rng(42)
    >>> blob = pd.DataFrame(rng.normal(size=(80, 3)))
    >>> fig = plot_manifold_embedding(manifold_embedding(blob, "mds"))
    >>> bool(fig.axes)
    True
    """
    try:
        coords = embedding.coords
        method = str(embedding.method)
        n_samples = int(embedding.n_samples)
    except (AttributeError, TypeError) as exc:
        raise DataValidationError(
            f"plot_manifold_embedding expects an EmbeddingRecord (from "
            f"manifold_embedding), got {type(embedding).__name__}"
        ) from exc
    if coords.shape[1] < 2:
        raise DataValidationError(
            "the embedding needs at least 2 dimensions to scatter "
            f"(got {coords.shape[1]})"
        )
    x = coords.iloc[:, 0].to_numpy(dtype=float)
    y = coords.iloc[:, 1].to_numpy(dtype=float)
    x_name, y_name = str(coords.columns[0]), str(coords.columns[1])

    fig, ax = new_figure(figsize=(7.8, 6.0))
    if labels is None:
        ax.scatter(x, y, s=26, alpha=0.75, color=PALETTE[0],
                   edgecolor="white", linewidth=0.3)
    else:
        if hasattr(labels, "to_numpy"):
            series = labels if isinstance(labels, pd.Series) else pd.Series(labels)
            if len(series) != n_samples:
                raise DataValidationError(
                    f"labels length ({len(series)}) must match the embedding rows ({n_samples})"
                )
            target = series.to_numpy()
        else:
            target = np.asarray(labels)
            if target.shape[0] != n_samples:
                raise DataValidationError(
                    f"labels length ({target.shape[0]}) must match the embedding rows ({n_samples})"
                )
        if pd.api.types.is_numeric_dtype(pd.Series(target)) and len(
            set(np.asarray(target, dtype=object).tolist())
        ) > 12:
            numeric = np.asarray(target, dtype=float)
            points = ax.scatter(x, y, s=26, alpha=0.8, c=numeric, cmap="viridis",
                                edgecolor="white", linewidth=0.3)
            fig.colorbar(points, ax=ax, shrink=0.85, pad=0.02)
        else:
            classes = sorted({str(value) for value in np.asarray(target, dtype=object)})
            lookup = {name: index for index, name in enumerate(classes)}
            codes = [lookup[str(value)] for value in np.asarray(target, dtype=object)]
            for index, name in enumerate(classes):
                mask = [code == index for code in codes]
                ax.scatter(
                    np.asarray(x)[mask], np.asarray(y)[mask], s=30, alpha=0.85,
                    color=PALETTE[index % len(PALETTE)],
                    edgecolor="white", linewidth=0.3, label=str(name),
                )
            ax.legend(fontsize=9, loc="best", title="class", title_fontsize=9)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    caption = title or f"{method} embedding ({n_samples} rows)"
    ax.set_title(caption, fontsize=11)
    return fig


def plot_hexbin_density(
    x: Any,
    y: Any,
    *,
    gridsize: int = 30,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "2-D density (hexbin)",
) -> Any:
    """Hexagonal binning -- the honest 2-D density for big scatter plots.

    A scatter of thousands of points becomes an ink blot; hexbin
    counts per cell and says so with a colorbar.  The house choice
    over scatter-overplotting for descriptor-pair views (PDSH 04.04 /
    04.05's ``hexbin`` section, absorbed).

    Parameters
    ----------
    x, y:
        Paired numeric columns (Series / arrays), equal length.
    gridsize:
        Hex cells across the x range (30 by default).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        Length mismatch, non-numeric input, or fewer than 10 pairs
        (below that a plain scatter tells the story better).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> frame = pd.DataFrame(rng.normal(size=(400, 2)), columns=["a", "b"])
    >>> fig = plot_hexbin_density(frame["a"], frame["b"])
    >>> bool(fig.axes)
    True
    """
    xs = np.asarray(
        x.iloc[:, 0].to_numpy(dtype=float) if isinstance(x, pd.DataFrame)
        else (x.to_numpy(dtype=float) if isinstance(x, pd.Series) else x),
        dtype=float,
    ).ravel()
    ys = np.asarray(
        y.iloc[:, 0].to_numpy(dtype=float) if isinstance(y, pd.DataFrame)
        else (y.to_numpy(dtype=float) if isinstance(y, pd.Series) else y),
        dtype=float,
    ).ravel()
    if xs.size != ys.size:
        raise DataValidationError(
            f"plot_hexbin_density: x and y must be the same length ({xs.size} vs {ys.size})"
        )
    if xs.size < 10:
        raise DataValidationError(
            f"plot_hexbin_density needs at least 10 pairs to be meaningful, got {xs.size}"
        )
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise DataValidationError("plot_hexbin_density: x/y must be finite (no NaN/±inf)")
    if gridsize < 4:
        raise DataValidationError(f"gridsize must be >= 4, got {gridsize}")

    fig, ax = new_figure(figsize=(7.8, 6.0))
    mesh = ax.hexbin(xs, ys, gridsize=int(gridsize), cmap="viridis", mincnt=1)
    fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label="pairs per cell")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{xs.size} pairs, {int(gridsize)} cells across", fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Optuna study figures (100x-160) — the seven single-objective visualizations
# of the optuna/optuna repo's ``visualization/matplotlib`` gallery, absorbed
# in the house dialect over the tuner's own records (TunerResults /
# TrialResult / HyperparameterImportance -- duck-typed, so the plotting layer
# never imports the tuner).
# ---------------------------------------------------------------------------
def _unwrap_trials(results: Any) -> tuple[list[Any], str]:
    """``(trials, objective label)`` from a TunerResults or a bare list.

    The duck-typed front door every study figure shares: a
    ``TunerResults`` (anything exposing ``.trials`` and ``.objective``)
    or a plain sequence of trial records (each carrying ``trial_id``,
    ``hyperparameters``, ``score`` and ``resolved_state``).  Anything
    else is the typed refusal.
    """
    trials: Any = None
    objective = "objective"
    if hasattr(results, "trials") and hasattr(results, "objective"):
        trials = list(results.trials)
        objective = str(results.objective)
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        trials = list(results)
    else:
        raise DataValidationError(
            "the study figures accept a TunerResults record or a list of "
            "TrialResult records, got "
            + type(results).__name__
        )
    if not trials:
        raise DataValidationError(
            "the study carries no trials -- run a search first "
            "(RandomSearch / TPESearch / Hyperband .search(X, y))"
        )
    for trial in trials:
        if not (hasattr(trial, "trial_id") and hasattr(trial, "score")):
            raise DataValidationError(
                "every trials entry must be a TrialResult record "
                "(trial_id / hyperparameters / score / resolved_state), got "
                + type(trial).__name__
            )
    return trials, objective


def _completed_trials(trials: list[Any]) -> list[Any]:
    """The resolved-'complete' trials with finite scores, in trial order."""
    return [
        t
        for t in trials
        if getattr(t, "resolved_state", "complete") == "complete"
        and bool(np.isfinite(float(t.score)))
    ]


def plot_optimization_history(
    results: Any,
    *,
    title: str = "Optimization history",
) -> Any:
    """The objective per trial + the best-so-far step line (100x-160).

    ``optuna.visualization.matplotlib.plot_optimization_history``'s
    figure: one dot per COMPLETED trial (x = trial number, y = its
    cross-validated objective), the best-so-far step line tying every
    running improvement together, and -- the house's honest addition
    over the upstream -- every PRUNED trial as an open amber marker at
    its last reported intermediate (the running fold mean the pruner
    acted on), so the pruner's own work is visible instead of hidden.

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    title:
        Chart title (the objective label rides the y axis).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form, carries no trials, or no trial
        resolved 'complete' with a finite score.

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(1, {"units": 8}, 0.5, 0.01),
    ...           TrialResult(2, {"units": 16}, 0.8, 0.01),
    ...           TrialResult(3, {"units": 32}, float("nan"), float("nan"),
    ...                       state="pruned", fold_scores=(0.1, 0.2))]
    >>> fig = plot_optimization_history(trials)
    >>> ax = fig.axes[0]
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('trial', 'objective')
    >>> len(ax.collections) >= 2          # completed + pruned markers
    True
    >>> len(ax.lines) >= 1                # the best-so-far step line
    True
    """
    trials, objective = _unwrap_trials(results)
    completed = _completed_trials(trials)
    if not completed:
        raise DataValidationError(
            "plot_optimization_history needs at least one completed trial "
            "with a finite score -- every trial failed or was pruned"
        )
    fig, ax = new_figure(figsize=(8.4, 5.2))
    ids = np.array([float(t.trial_id) for t in completed])
    scores = np.array([float(t.score) for t in completed])
    ax.scatter(ids, scores, s=34, color=PALETTE[0], alpha=0.85, zorder=3,
               label="completed trial")
    best = np.maximum.accumulate(scores)
    ax.step(ids, best, where="post", color=PALETTE[2], lw=1.6, zorder=2,
            label="best so far")
    pruned = [t for t in trials if getattr(t, "resolved_state", "") == "pruned"]
    if pruned:
        pruned_ids, pruned_values = [], []
        for trial in pruned:
            folds = trial.fold_scores
            if folds:
                pruned_ids.append(float(trial.trial_id))
                pruned_values.append(float(np.asarray(folds, dtype=float)[-1]))
        if pruned_ids:
            ax.scatter(pruned_ids, pruned_values, s=30, facecolors="none",
                       edgecolors=PALETTE[1], lw=1.4, zorder=3,
                       label="pruned (last intermediate)")
    ax.set_xlabel("trial")
    ax.set_ylabel(objective)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)
    return fig


def plot_slice(
    results: Any,
    *,
    params: Sequence[str] | None = None,
    title: str = "Slice plot",
) -> Any:
    """One param-vs-objective panel per hyperparameter (100x-160).

    ``optuna.visualization.matplotlib.plot_slice``'s figure: the
    objective (y) against each hyperparameter's value (x), one subplot
    per param, completed trials only (a pruned or failed trial owns no
    full-budget score to place).  Numeric params keep their numeric
    axis; categorical / boolean params get category-index positions
    with their values as tick labels.  ``params=None`` shows every
    hyperparameter the trials declare (the first completed trial's
    declaration order).

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    params:
        The hyperparameter names to show; ``None`` shows all.  An
        unknown name is the typed refusal.
    title:
        Chart title (the objective label rides the y axes).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form, no trial resolved 'complete',
        or ``params`` names a hyperparameter no completed trial
        declares.

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(1, {"units": 8, "alpha": 0.1}, 0.5, 0.01),
    ...           TrialResult(2, {"units": 16, "alpha": 0.2}, 0.8, 0.01)]
    >>> fig = plot_slice(trials)
    >>> len(fig.axes)                     # one panel per param
    2
    >>> fig.axes[0].get_xlabel()
    'units'
    """
    trials, objective = _unwrap_trials(results)
    completed = _completed_trials(trials)
    if not completed:
        raise DataValidationError(
            "plot_slice needs at least one completed trial with a finite "
            "score -- every trial failed or was pruned"
        )
    declared = list(dict.fromkeys(
        key for trial in completed for key in trial.hyperparameters
    ))
    chosen = declared if params is None else [str(p) for p in params]
    if not chosen:
        raise DataValidationError(
            "params must name at least one hyperparameter the trials declare"
        )
    unknown = [name for name in chosen if name not in declared]
    if unknown:
        raise DataValidationError(
            f"params names hyperparameter(s) no completed trial declares: "
            f"{', '.join(unknown)} -- declared: {', '.join(declared)}"
        )
    scores = np.array([float(t.score) for t in completed])
    n_cols = 2 if len(chosen) > 1 else 1
    n_rows = (len(chosen) + n_cols - 1) // n_cols
    fig, axes = new_figure(
        nrows=n_rows, ncols=n_cols, figsize=(4.4 * n_cols, 3.4 * n_rows + 0.8)
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes_array, chosen):
        raw = [trial.hyperparameters[name] for trial in completed]
        numeric = all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            for value in raw
        )
        if numeric:
            xs = np.array([float(value) for value in raw])
            ax.scatter(xs, scores, s=30, color=PALETTE[0], alpha=0.85)
            ax.set_xlabel(name)
        else:
            categories = list(dict.fromkeys(str(value) for value in raw))
            positions = {category: float(i) for i, category in enumerate(categories)}
            xs = np.array([positions[str(value)] for value in raw])
            ax.scatter(xs, scores, s=30, color=PALETTE[0], alpha=0.85)
            ax.set_xticks(list(positions.values()),
                          list(positions.keys()), fontsize=8)
            ax.set_xlabel(name)
        ax.set_ylabel(objective)
        ax.tick_params(labelsize=8)
    for ax in axes_array[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_parallel_coordinate(
    results: Any,
    *,
    params: Sequence[str] | None = None,
    title: str = "Parallel coordinates",
) -> Any:
    """One polyline per trial across the params + objective (100x-160).

    ``optuna.visualization.matplotlib.plot_parallel_coordinate``'s
    figure: every completed trial as a polyline whose first axes are
    the hyperparameters (each min-max normalised to [0, 1], tick
    labels carrying the real values) and whose LAST axis is the
    objective itself; the line colour is the trial's score through the
    house amber->teal ramp, so good and bad configurations separate
    at a glance.  Categorical / boolean params use category indices.

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    params:
        The hyperparameter names to show; ``None`` shows all.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form, or fewer than TWO trials
        resolved 'complete' with finite scores (a single line has no
        comparative story -- the honest refusal).

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(1, {"units": 8}, 0.5, 0.01),
    ...           TrialResult(2, {"units": 16}, 0.8, 0.01),
    ...           TrialResult(3, {"units": 32}, 0.65, 0.01)]
    >>> fig = plot_parallel_coordinate(trials)
    >>> ax = fig.axes[0]
    >>> ax.get_xticklabels()[0].get_text().split("=")[0]
    'units'
    """
    trials, objective = _unwrap_trials(results)
    completed = _completed_trials(trials)
    if len(completed) < 2:
        raise DataValidationError(
            f"plot_parallel_coordinate needs at least two completed trials "
            f"with finite scores to compare, got {len(completed)}"
        )
    declared = list(dict.fromkeys(
        key for trial in completed for key in trial.hyperparameters
    ))
    chosen = declared if params is None else [str(p) for p in params]
    if not chosen:
        raise DataValidationError(
            "params must name at least one hyperparameter the trials declare"
        )
    unknown = [name for name in chosen if name not in declared]
    if unknown:
        raise DataValidationError(
            f"params names hyperparameter(s) no completed trial declares: "
            f"{', '.join(unknown)} -- declared: {', '.join(declared)}"
        )
    scores = np.array([float(t.score) for t in completed])
    columns: list[tuple[str, np.ndarray, list[str] | None, float, float]] = []
    for name in chosen:
        raw = [trial.hyperparameters[name] for trial in completed]
        numeric = all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            for value in raw
        )
        if numeric:
            values = np.array([float(value) for value in raw])
            columns.append((name, values, None, float(values.min()), float(values.max())))
        else:
            category_list = list(dict.fromkeys(str(value) for value in raw))
            positions = {category: float(i) for i, category in enumerate(category_list)}
            values = np.array([positions[str(value)] for value in raw])
            columns.append((name, values, category_list, 0.0, float(len(category_list) - 1)))
    lo, hi = float(scores.min()), float(scores.max())

    def normalized(values: np.ndarray, low: float, high: float) -> np.ndarray:
        span = high - low
        if span <= 0.0:
            return np.full(values.shape, 0.5)
        return (values - low) / span

    fig, ax = new_figure(figsize=(8.8, 5.4))
    xs = np.arange(len(columns) + 1)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list(
        "pharmaml_score", [PALETTE[1], "#FFFFFF", PALETTE[0]]
    )
    colour_values = normalized(scores, lo, hi)
    for row_index, trial in enumerate(completed):
        ys = [normalized(column[1], column[3], column[4])[row_index] for column in columns]
        ys.append(colour_values[row_index])
        ax.plot(xs, ys, lw=1.3, alpha=0.75, color=cmap(colour_values[row_index]))
    # the range (or the categories) of every param rides BELOW its axis --
    # the normalised positions alone would be unreadable.
    for position, (name, _values, categories, low, high) in enumerate(columns):
        footer = (
            f"{name}: " + " | ".join(categories)
            if categories is not None
            else f"{name}: {low:g} … {high:g}"
        )
        ax.text(position, -0.09, footer, ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(xs)
    ax.set_xticklabels([column[0] for column in columns] + [objective], fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.25, len(xs) - 0.75)
    ax.set_ylabel("normalised value")
    ax.set_title(title, fontsize=11)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=lo, vmax=hi))
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label(objective)
    return fig


def plot_param_importances(
    importances: Any,
    *,
    top_n: int | None = None,
    title: str = "Hyperparameter importances",
) -> Any:
    """The MDI hyperparameter-importance bars (100x-160).

    ``optuna.visualization.matplotlib.plot_param_importances``'s
    figure: one horizontal bar per hyperparameter, the random forest's
    normalised mean-decrease-in-impurity attribution, largest at the
    top, the value labelled at the bar's end.  Accepts the
    :class:`~pharmaml.ml.tuner.HyperparameterImportance` record OR its
    ``as_frame`` DataFrame (``parameter`` / ``importance`` columns) --
    the same dual-form contract as :func:`plot_feature_importance`.

    Parameters
    ----------
    importances:
        A :class:`~pharmaml.ml.tuner.HyperparameterImportance` record,
        or a DataFrame with ``parameter`` / ``importance`` columns.
    top_n:
        How many params to show (largest first); ``None`` shows all.
    title:
        Chart title (the method rides it).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``importances`` carries neither the record attributes nor
        the DataFrame columns.

    Examples
    --------
    >>> from pharmaml.ml.tuner import HyperparameterImportance
    >>> record = HyperparameterImportance(
    ...     names=("lr", "depth"), importances=(0.8, 0.2), n_trials=6,
    ...     objective="r2")
    >>> fig = plot_param_importances(record)
    >>> [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    ['depth', 'lr']
    """
    _check_top_n(top_n)
    if hasattr(importances, "names") and hasattr(importances, "importances"):
        names = [str(name) for name in importances.names]
        values = np.asarray(importances.importances, dtype=float)
        method = str(getattr(importances, "method", "mdi"))
    elif isinstance(importances, pd.DataFrame) and {
        "parameter", "importance"
    }.issubset(importances.columns):
        names = [str(name) for name in importances["parameter"].tolist()]
        values = importances["importance"].to_numpy(dtype=float)
        method = "importance"
    else:
        raise DataValidationError(
            "importances must be a HyperparameterImportance record "
            "(names/importances) or a DataFrame with parameter/importance "
            f"columns, got {type(importances).__name__}"
        )
    if len(names) != len(values):
        raise DataValidationError(
            f"names carries {len(names)} label(s) for {len(values)} importance(s)"
        )
    keep = len(values) if top_n is None else min(int(top_n), len(values))
    order = np.argsort(-values, kind="stable")[:keep][::-1]

    fig, ax = new_figure(figsize=(8.4, 0.5 * len(order) + 1.8))
    positions = np.arange(len(order))
    ax.barh(positions, values[order], color=PALETTE[0], alpha=0.88,
            height=0.62, edgecolor="white")
    for pos, value in zip(positions, values[order]):
        ax.text(value, pos, f" {value:.4g}", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, [names[int(index)] for index in order])
    ax.set_xlabel(f"importance ({method})")
    ax.set_title(title, fontsize=11)
    return fig


def plot_edf(
    results: Any,
    *,
    title: str = "Objective EDF",
) -> Any:
    """The empirical CDF of the completed trials' scores (100x-160).

    ``optuna.visualization.matplotlib.plot_edf``'s figure: the
    empirical distribution function of the objective over the
    completed trials -- x = the objective value, y = the fraction of
    trials at or below it, drawn as the step curve with the quartile
    markers (Q1 / median / Q3) annotated.  The figure the "how spread
    are my trials, where does the good tail start" question asks.

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form or no trial resolved 'complete'
        with a finite score.

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(i, {"units": 8 * i}, float(i) / 4.0, 0.01)
    ...           for i in range(1, 6)]
    >>> fig = plot_edf(trials)
    >>> ax = fig.axes[0]
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('objective', 'fraction of trials ≤ value')
    """
    trials, objective = _unwrap_trials(results)
    completed = _completed_trials(trials)
    if not completed:
        raise DataValidationError(
            "plot_edf needs at least one completed trial with a finite "
            "score -- every trial failed or was pruned"
        )
    scores = np.sort(np.array([float(t.score) for t in completed]))
    fraction = np.arange(1, scores.size + 1) / scores.size

    fig, ax = new_figure(figsize=(7.6, 5.2))
    ax.step(scores, fraction, where="post", color=PALETTE[0], lw=1.8)
    ax.scatter(scores, fraction, s=26, color=PALETTE[0], alpha=0.85, zorder=3)
    for quantile, label in ((25, "Q1"), (50, "median"), (75, "Q3")):
        value = float(np.percentile(scores, quantile))
        level = float(np.mean(scores <= value))
        ax.axvline(value, color=PALETTE[2], lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            f"{label} = {value:.4g}", xy=(value, level), xytext=(6, 0),
            textcoords="offset points", fontsize=8, va="center",
            color=PALETTE[2],
        )
    ax.set_xlabel(objective)
    ax.set_ylabel("fraction of trials ≤ value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title} — {scores.size} completed trial(s)", fontsize=11)
    return fig


def plot_timeline(
    results: Any,
    *,
    title: str = "Trial timeline",
) -> Any:
    """Per-trial duration bars coloured by state (100x-160).

    ``optuna.visualization.matplotlib.plot_timeline``'s figure,
    honestly translated: the upstream plots wall-clock start->end
    stamps of possibly-parallel trials; the toolkit's searches are
    strictly sequential, so the honest twin is a stacked-duration
    Gantt -- trial i's bar starts where trial i-1's ended, the bar's
    WIDTH is the trial's recorded ``duration`` seconds, and the colour
    is the resolved state (complete = teal, pruned = amber, fail =
    red).  Trials without a recorded duration are excluded and named
    in the title's suffix line.

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    title:
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form, or no trial carries a recorded
        duration (the cross_val_score fast path records none -- run a
        search with a pruner to get durations).

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(1, {"units": 8}, 0.5, 0.01, duration=1.5),
    ...           TrialResult(2, {"units": 16}, float("nan"), float("nan"),
    ...                       state="pruned", duration=0.5)]
    >>> fig = plot_timeline(trials)
    >>> ax = fig.axes[0]
    >>> ax.get_xlabel()
    'cumulative seconds'
    """
    trials, _objective = _unwrap_trials(results)
    timed = [t for t in trials if getattr(t, "duration", None) is not None]
    if not timed:
        raise DataValidationError(
            "plot_timeline needs trials with a recorded duration -- none of "
            f"the {len(trials)} trial(s) carry one (the cross_val_score fast "
            "path records none; a pruner-wired search does)"
        )
    excluded = len(trials) - len(timed)
    state_colours = {"complete": PALETTE[0], "pruned": PALETTE[1], "fail": PALETTE[3]}
    state_labels = {"complete": "complete", "pruned": "pruned", "fail": "fail"}

    fig, ax = new_figure(figsize=(8.6, 0.42 * len(timed) + 1.8))
    positions = np.arange(len(timed))[::-1]
    cursor = 0.0
    present: dict[str, str] = {}
    for position, trial in zip(positions, timed):
        duration = float(trial.duration or 0.0)
        state = getattr(trial, "resolved_state", "complete")
        colour = state_colours.get(state, PALETTE[2])
        present.setdefault(state, state_labels.get(state, state))
        ax.barh(position, duration, left=cursor, height=0.6, color=colour,
                alpha=0.88, edgecolor="white")
        ax.text(cursor + duration, position, f" {duration:.2f}s", va="center",
                ha="left", fontsize=8, color=PALETTE[2])
        cursor += duration
    ax.set_yticks(positions, [f"trial {t.trial_id}" for t in timed])
    ax.set_xlabel("cumulative seconds")
    suffix = f" — {excluded} without a recorded duration excluded" if excluded else ""
    ax.set_title(title + suffix, fontsize=11)
    if present:
        from matplotlib.patches import Patch as plt_Patch

        handles = [
            plt_Patch(color=state_colours[state], label=label)
            for state, label in present.items()
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8)
    return fig


def plot_intermediate_values(
    results: Any,
    *,
    title: str = "Intermediate values",
) -> Any:
    """The per-trial running fold-mean curves (100x-160).

    ``optuna.visualization.matplotlib.plot_intermediate_values``'s
    figure: one curve per trial that reported intermediates -- for the
    toolkit that is every pruner-wired trial, whose reported values
    are the RUNNING mean of the fold scores after each fold (exactly
    what :class:`~pharmaml.ml.tuner.PercentilePruner` acted on).
    Completed trials draw as solid teal lines with circle markers;
    pruned trials as dashed amber lines ending at their prune fold;
    the x axis is the 1-based fold index.

    Parameters
    ----------
    results:
        A :class:`~pharmaml.ml.tuner.TunerResults` record or a list of
        :class:`~pharmaml.ml.tuner.TrialResult` records.
    title:
        Chart title (the objective label rides the y axis).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``results`` is neither form, or no trial carries
        ``fold_scores`` (the cross_val_score fast path records none --
        run a search with a pruner to get intermediates).

    Examples
    --------
    >>> from pharmaml.ml.tuner import TrialResult
    >>> trials = [TrialResult(1, {"units": 8}, 0.5, 0.01,
    ...                       fold_scores=(0.4, 0.5, 0.6)),
    ...           TrialResult(2, {"units": 16}, float("nan"), float("nan"),
    ...                       state="pruned", fold_scores=(0.1, 0.2))]
    >>> fig = plot_intermediate_values(trials)
    >>> ax = fig.axes[0]
    >>> ax.get_xlabel()
    'fold (running mean)'
    """
    trials, objective = _unwrap_trials(results)
    reporting = [
        t for t in trials
        if getattr(t, "fold_scores", None) is not None and len(t.fold_scores) > 0
    ]
    if not reporting:
        raise DataValidationError(
            "plot_intermediate_values needs trials with fold_scores -- none "
            f"of the {len(trials)} trial(s) carry them (the cross_val_score "
            "fast path records none; a pruner-wired search does)"
        )
    fig, ax = new_figure(figsize=(8.4, 5.2))
    present: set[str] = set()
    for trial in reporting:
        folds = np.asarray(trial.fold_scores, dtype=float)
        running = np.cumsum(folds) / np.arange(1, folds.size + 1)
        steps = np.arange(1, folds.size + 1)
        state = getattr(trial, "resolved_state", "complete")
        present.add(state)
        if state == "pruned":
            ax.plot(steps, running, ls="--", color=PALETTE[1], lw=1.5,
                    marker="o", ms=4, alpha=0.9,
                    label=f"trial {trial.trial_id} (pruned)")
        else:
            ax.plot(steps, running, ls="-", color=PALETTE[0], lw=1.5,
                    marker="o", ms=4, alpha=0.9,
                    label=f"trial {trial.trial_id}")
    ax.set_xlabel("fold (running mean)")
    ax.set_ylabel(objective)
    ax.set_title(title, fontsize=11)
    if len(present) > 1 or len(reporting) <= 8:
        ax.legend(fontsize=8, loc="best")
    else:
        from matplotlib.lines import Line2D as plt_Line2D

        ax.legend([plt_Line2D([], [], color=PALETTE[0], marker="o", ms=4)],
                  ["completed trial"], fontsize=8, loc="best")
    return fig

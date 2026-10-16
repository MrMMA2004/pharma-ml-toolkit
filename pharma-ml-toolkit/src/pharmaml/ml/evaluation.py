"""Model evaluation with structured, report-friendly result objects.

The legacy scripts printed ``classification_report`` to stdout and
immediately dumped PNG/PDF files; this module returns typed report
objects (dataclasses) that callers can print, serialise, or feed into
report generators -- printing is a one-line convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
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

__all__ = ["ClassificationReport", "RegressionReport", "evaluate_model"]


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
        On invalid task, empty ``y_test``, or prediction-length
        mismatch.
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

    if task == "classification":
        labels = sorted(np.unique(np.concatenate([y_true, y_pred]).tolist()))
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    roc_auc = float(roc_auc_score(y_true, proba[:, 1]))
                else:
                    roc_auc = float(
                        roc_auc_score(y_true, proba, multi_class="ovr", labels=labels)
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

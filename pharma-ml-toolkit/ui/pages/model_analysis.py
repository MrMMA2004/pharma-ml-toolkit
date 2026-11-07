"""Model Analysis: real training, real curves, off the GUI thread (16.8.6/16.14).

Pick a bundled QSAR dataset (or the table opened in Data Explorer), a
target column and a registry model; ``train_model`` runs in a
:class:`JobWorker` (QThreadPool) and the finished page plots REAL
predictions-vs-actual, residuals and -- only when the estimator
exposes them -- feature importances. Metrics come from the actual
held-out split; nothing on this page is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]
_DATASETS = {
    "qsar_regression.csv": "pIC50",
    "qsar_classification.csv": "activity",
    "walkthrough_classification.csv": "",
}


@dataclass
class AnalysisOutcome:
    """What the background job hands back (all real numbers)."""

    task: str = ""
    model: str = ""
    n_train: int = 0
    n_test: int = 0
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    metric_name: str = ""
    metric_value: float = float("nan")
    cv_mean: float = float("nan")
    importance_names: list[str] = field(default_factory=list)
    importance_values: np.ndarray | None = None
    seconds: float = 0.0


class ModelAnalysisPage(QWidget):
    """Train a registry model on a real dataset and inspect the fit."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        self._frame: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._dataset_combo = QComboBox()
        for name in _DATASETS:
            self._dataset_combo.addItem(name)
        self._dataset_combo.addItem("(table from Data Explorer)")
        self._dataset_combo.currentTextChanged.connect(self._load_dataset)
        self._target_combo = QComboBox()
        self._model_combo = QComboBox()
        try:
            from pharmaml.ml.models import MODEL_REGISTRY

            for key in sorted(MODEL_REGISTRY):
                self._model_combo.addItem(key)
            self._model_combo.setCurrentText("random_forest")
        except Exception:  # noqa: BLE001 - broken-install fallback to the default entry
            self._model_combo.addItem("random_forest")
        self._train_btn = QPushButton("Train")
        self._train_btn.setMinimumHeight(34)
        self._train_btn.clicked.connect(self._start_training)
        form.addRow("Dataset:", self._dataset_combo)
        form.addRow("Target column:", self._target_combo)
        form.addRow("Model:", self._model_combo)
        form.addRow("", self._train_btn)
        lay.addLayout(form)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pred_plot = PlotPanel(title="Predicted vs actual (held-out split)")
        self._resid_plot = PlotPanel(title="Residuals vs predicted")
        splitter.addWidget(self._pred_plot)
        splitter.addWidget(self._resid_plot)
        splitter.setSizes([420, 420])
        lay.addWidget(splitter, stretch=1)

        self._importance_plot = PlotPanel(title="Feature importance (when the estimator exposes it)")
        self._importance_plot.setMinimumHeight(220)
        lay.addWidget(self._importance_plot)
        self._metric_label = QLabel("No training run yet.")
        self._metric_label.setObjectName("studioStatus")
        self._metric_label.setWordWrap(True)
        lay.addWidget(self._metric_label)

        self._load_dataset(self._dataset_combo.currentText())

    # -- data ---------------------------------------------------------------
    def _load_dataset(self, name: str) -> None:
        self._target_combo.clear()
        if name == "(table from Data Explorer)":
            frame = None
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                frame = self._data_explorer.frame()
            if frame is None or frame.empty:
                self._frame = None
                self._metric_label.setText("Data Explorer has no table loaded yet.")
                return
            self._frame = frame
        else:
            path = _TK / "examples" / "data" / name
            if not path.exists():  # pragma: no cover - missing example
                self._frame = None
                return
            from pharmaml.utils.io import read_table

            self._frame = read_table(path)
        default = _DATASETS.get(name, "")
        for col in self._frame.columns:
            self._target_combo.addItem(str(col))
        if default and default in [str(c) for c in self._frame.columns]:
            self._target_combo.setCurrentText(default)

    # -- training ------------------------------------------------------------
    def _on_started(self, name: str) -> None:
        self._train_btn.setEnabled(False)
        self._metric_label.setText(f"Training {name}…")

    def _on_ended(self) -> None:
        self._train_btn.setEnabled(True)

    def _start_training(self) -> None:
        if self._frame is None:
            self._metric_label.setText("Load a dataset first.")
            return
        frame = self._frame
        target = self._target_combo.currentText()
        model = self._model_combo.currentText()
        if not target or target not in frame.columns:
            self._metric_label.setText("Pick a target column.")
            return
        if self._runner.busy:
            self._metric_label.setText("A job is already running — Stop it first.")
            return

        y = frame[target]
        numeric_y = pd.api.types.is_numeric_dtype(y)
        classes = y.astype(str).nunique()
        task = "regression" if numeric_y and classes > 10 else "classification"
        X = frame.drop(columns=[target])
        X = X.select_dtypes(include=[np.number])

        from sklearn.model_selection import train_test_split

        stratify = None if task == "regression" else y.astype(str).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )

        def job(progress) -> AnalysisOutcome:
            from pharmaml.ml.training import train_model

            progress(0.1, "loading")
            result = train_model(
                model,
                task,
                X_train,
                y_train,
                tune=False,
                preprocess=True,
                n_jobs=1,
            )
            progress(0.7, "predicting")
            fitted = result.model
            y_pred = fitted.predict(X_test)
            outcome = AnalysisOutcome(
                task=task,
                model=model,
                n_train=len(X_train),
                n_test=len(X_test),
                y_test=np.asarray(y_test, dtype=float),
                y_pred=np.asarray(y_pred, dtype=float),
                cv_mean=(
                    float(result.cv_mean)
                    if result.cv_mean is not None and result.cv_mean == result.cv_mean
                    else float("nan")
                ),
                seconds=float(result.fit_seconds),
            )
            progress(0.85, "scoring")
            if task == "regression":
                from sklearn.metrics import r2_score

                outcome.metric_name = "R² (held-out)"
                outcome.metric_value = float(r2_score(outcome.y_test, outcome.y_pred))
            else:
                from sklearn.metrics import accuracy_score

                outcome.metric_name = "accuracy (held-out)"
                outcome.metric_value = float(accuracy_score(np.asarray(y_test).astype(str), np.asarray(y_pred).astype(str)))
            estimator = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
            names = list(X.columns)
            if hasattr(estimator, "feature_importances_"):
                outcome.importance_names = names
                outcome.importance_values = np.asarray(estimator.feature_importances_, dtype=float)
            elif hasattr(estimator, "coef_"):
                coef = np.asarray(estimator.coef_, dtype=float)
                if coef.ndim == 1:
                    outcome.importance_names = names
                    outcome.importance_values = np.abs(coef)
            progress(1.0, "done")
            return outcome

        worker = self._runner.submit(f"{model} on {target}", job)
        if worker is None:  # pragma: no cover - busy guard already checked
            return
        worker.signals.finished.connect(self._render)
        worker.signals.failed.connect(self._failed)
        worker.signals.cancelled.connect(self._cancelled)
        worker.signals.progress.connect(self._progress.update_progress)

    # -- rendering --------------------------------------------------------------
    def _render(self, outcome: AnalysisOutcome) -> None:
        self._pred_plot.clear()
        self._resid_plot.clear()
        self._importance_plot.clear()
        o: AnalysisOutcome = outcome
        if o.y_test is None or o.y_pred is None:
            return
        if o.task == "regression":
            lo = float(min(o.y_test.min(), o.y_pred.min()))
            hi = float(max(o.y_test.max(), o.y_pred.max()))
            span = np.array([lo, hi])
            self._pred_plot.line(span, span, label="y = x", width=1, index=1, style="dash")
            self._pred_plot.scatter(o.y_test, o.y_pred, label="held-out samples")
            self._pred_plot.set_labels(x="actual", y="predicted")
            self._resid_plot.scatter(o.y_pred, o.y_test - o.y_pred)
            self._resid_plot.line(
                np.array([float(o.y_pred.min()), float(o.y_pred.max())]),
                np.zeros(2),
                width=1,
                index=1,
                style="dash",
            )
            self._resid_plot.set_labels(x="predicted", y="residual")
        else:
            classes = sorted(set(np.asarray(o.y_test).astype(str)) | set(np.asarray(o.y_pred).astype(str)))
            matrix = np.zeros((len(classes), len(classes)), dtype=float)
            for actual, predicted in zip(np.asarray(o.y_test).astype(str), np.asarray(o.y_pred).astype(str)):
                matrix[classes.index(actual), classes.index(predicted)] += 1
            self._pred_plot.bars(matrix.sum(axis=1), labels=classes)
            self._pred_plot.set_labels(x="class", y="count", title="Actual class counts")
            self._importance_plot.set_labels(x="feature", y="|coefficient| / importance")
        if o.importance_values is not None and len(o.importance_names):
            order = np.argsort(o.importance_values)[::-1][:12]
            self._importance_plot.bars(
                o.importance_values[order],
                labels=[o.importance_names[i] for i in order],
            )
            self._importance_plot.set_labels(x="feature", y="importance")
        cv = f" · CV mean {o.cv_mean:.3f}" if o.cv_mean == o.cv_mean else ""
        self._metric_label.setText(
            f"{o.model} ({o.task}) · train {o.n_train} / test {o.n_test} · "
            f"{o.metric_name} {o.metric_value:.3f}{cv} · fitted in {o.seconds:.1f}s"
        )

    def _failed(self, message: str) -> None:
        self._metric_label.setText(f"Training failed — {message.splitlines()[0]}")

    def _cancelled(self, seconds: float) -> None:
        self._metric_label.setText(f"Training cancelled after {seconds:.1f}s (cooperative stop).")

    def apply_palette(self, palette: ChartPalette) -> None:
        for plot in (self._pred_plot, self._resid_plot, self._importance_plot):
            plot.apply_palette(palette)

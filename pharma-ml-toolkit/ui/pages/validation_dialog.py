"""Nested regression validation with honest fold-level progress and stop."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class ValidationDialog(QDialog):
    """Nested ridge CV; stopping is cooperative at outer fold boundaries only."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nested ridge validation")
        self.resize(730, 650)
        self._runner = runner
        self._revision = 0
        self._running_own = False
        layout = QVBoxLayout(self)
        description = QLabel("Nested CV evaluates a fixed ridge tuning procedure, not a final fitted model. "
                             "Choose leakage-unit groups whenever rows are related.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        row = QHBoxLayout()
        self._data = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self._data, 1)
        row.addWidget(browse)
        form.addRow("Dataset:", row)
        self._features = QPlainTextEdit()
        self._features.setPlaceholderText("One numeric feature column per line")
        self._features.setMaximumHeight(80)
        form.addRow("Features:", self._features)
        self._target, self._group = QLineEdit(), QLineEdit()
        self._group.setPlaceholderText("Optional leakage-unit group column")
        form.addRow("Target:", self._target)
        form.addRow("Group:", self._group)
        self._outer, self._inner, self._seed = QSpinBox(), QSpinBox(), QSpinBox()
        for control in (self._outer, self._inner):
            control.setRange(2, 20)
            control.setValue(3)
        self._seed.setRange(0, 2_147_483_000)
        self._seed.setValue(42)
        form.addRow("Outer folds:", self._outer)
        form.addRow("Inner folds:", self._inner)
        form.addRow("Seed (row split only):", self._seed)
        layout.addLayout(form)
        self._controls: tuple[QWidget, ...] = (self._data, browse, self._features, self._target,
                                                self._group, self._outer, self._inner, self._seed)
        for field in (self._data, self._target, self._group):
            field.textChanged.connect(self._invalidate)
        self._features.textChanged.connect(self._invalidate)
        for spin in (self._outer, self._inner, self._seed):
            spin.valueChanged.connect(self._invalidate)
        buttons = QHBoxLayout()
        self._run = QPushButton("Run nested validation")
        self._run.clicked.connect(self._start)
        self._stop = QPushButton("Stop at next fold boundary")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._request_stop)
        buttons.addWidget(self._run)
        buttons.addWidget(self._stop)
        layout.addLayout(buttons)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
        self._status = QLabel("No validation run yet. This operation cannot be resumed.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Preprocessing and tuning are inside each fold. Without a group column, "
                         "random row folds can leak related compounds. Stop cannot interrupt an active fold; "
                         "cancelled partial results are not saved or resumable.")
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset", "",
                                             "Tables (*.csv *.tsv *.txt *.xlsx *.xlsm);;All files (*)")
        if path:
            self._data.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        if hasattr(self, "_result"):
            self._result.clear()
            self._progress.setValue(0)
            self._status.setText("Inputs changed; run validation again.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)
        self._run.setEnabled(enabled)
        self._stop.setEnabled(not enabled)

    def _start(self) -> None:
        from pharmaml.ml.validation_workbench import nested_ridge_from_file

        path = self._data.text().strip()
        features = [line.strip() for line in self._features.toPlainText().splitlines() if line.strip()]
        target, group = self._target.text().strip(), self._group.text().strip() or None
        outer, inner, seed = self._outer.value(), self._inner.value(), self._seed.value()
        if not path or not features or not target:
            self._result.clear()
            self._status.setText("Select a dataset and specify features and target.")
            return
        revision = self._revision
        self._result.clear()
        self._progress.setValue(0)
        self._status.setText("Running nested validation; Stop takes effect at a fold boundary.")
        self._enable(False)
        self._running_own = True

        def run(progress):
            return nested_ridge_from_file(
                Path(path), features=features, target=target, group=group,
                outer_folds=outer, inner_folds=inner, seed=seed,
                on_fold=lambda done, total: progress(done / total, f"Completed {done}/{total} outer folds"),
            )

        worker = self._runner.submit("Nested validation", run,
                                     on_progress=self._progress_update,
                                     on_finished=lambda result: self._finished(result, revision),
                                     on_failed=lambda error: self._failed(error, revision),
                                     on_cancelled=lambda elapsed: self._cancelled(elapsed, revision))
        if worker is None:
            self._running_own = False
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _request_stop(self) -> None:
        if self._running_own:
            self._runner.request_stop()
            self._stop.setEnabled(False)
            self._status.setText("Stop requested; waiting for the current outer fold to finish.")

    def _progress_update(self, percent: int, label: str) -> None:
        self._progress.setValue(percent)
        if self._stop.isEnabled():
            self._status.setText(f"{label}; Stop takes effect at a fold boundary.")

    def _finished(self, result: object, revision: int) -> None:
        self._running_own = False
        self._enable(True)
        if revision == self._revision:
            self._progress.setValue(100)
            self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
            self._status.setText("Nested validation complete; review fold scores and limitations.")

    def _failed(self, error: str, revision: int) -> None:
        self._running_own = False
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Validation failed: {error.splitlines()[0]}")

    def _cancelled(self, _seconds: float, revision: int) -> None:
        self._running_own = False
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText("Cancelled at a fold boundary. Partial results were not saved; no resume is available.")

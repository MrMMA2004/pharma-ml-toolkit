"""Held-out uncertainty using three explicitly selected tables."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class UncertaintyDialog(QDialog):
    """Fit a fixed model only on train, calibrate and evaluate on separate files."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Held-out uncertainty")
        self.resize(760, 700)
        self._runner = runner
        self._revision = 0
        self._last_report: object | None = None
        layout = QVBoxLayout(self)
        intro = QLabel("Select separate, independent train, calibration and test files. "
                       "This fits a new model; it does not reuse the current trained model.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self._paths: dict[str, QLineEdit] = {}
        self._controls: list[QWidget] = []
        self._task = QComboBox()
        self._task.addItem("Regression — ridge intervals", "regression")
        self._task.addItem("Classification — logistic prediction sets", "classification")
        self._task.currentIndexChanged.connect(self._invalidate)
        form.addRow("Task:", self._task)
        self._controls.append(self._task)
        for name in ("train", "calibration", "test"):
            row = QHBoxLayout()
            field = QLineEdit()
            field.setPlaceholderText(f"{name} CSV, TSV, XLSX or XLSM")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, edit=field: self._browse(edit))
            field.textChanged.connect(self._invalidate)
            row.addWidget(field, 1)
            row.addWidget(browse)
            form.addRow(f"{name.title()}:", row)
            self._paths[name] = field
            self._controls.extend((field, browse))
        self._features = QPlainTextEdit()
        self._features.setPlaceholderText("One numeric feature column name per line")
        self._features.setMaximumHeight(80)
        self._features.textChanged.connect(self._invalidate)
        form.addRow("Features:", self._features)
        self._controls.append(self._features)
        self._target = QLineEdit()
        self._key = QLineEdit()
        self._group = QLineEdit()
        self._group.setPlaceholderText("Optional, prespecified test subgroup")
        for label, field in (("Target:", self._target), ("Leakage-unit ID:", self._key),
                             ("Subgroup:", self._group)):
            field.textChanged.connect(self._invalidate)
            form.addRow(label, field)
            self._controls.append(field)
        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.001, 0.999)
        self._alpha.setDecimals(3)
        self._alpha.setSingleStep(0.01)
        self._alpha.setValue(0.1)
        self._alpha.valueChanged.connect(self._invalidate)
        form.addRow("Alpha:", self._alpha)
        self._controls.append(self._alpha)
        self._diagnostics = QCheckBox("Include kNN distance and classification probability diagnostics")
        self._diagnostics.setToolTip("Optional; kNN can be expensive. Diagnostics are descriptive, not a safety guarantee.")
        self._diagnostics.stateChanged.connect(self._invalidate)
        form.addRow("Diagnostics:", self._diagnostics)
        self._controls.append(self._diagnostics)
        layout.addLayout(form)
        self._run = QPushButton("Evaluate independent partitions")
        self._run.clicked.connect(self._start)
        buttons = QHBoxLayout()
        buttons.addWidget(self._run)
        self._save = QPushButton("Save report…")
        self._save.setEnabled(False)
        self._save.clicked.connect(self._save_report)
        buttons.addWidget(self._save)
        self._compare = QPushButton("Compare saved reports…")
        self._compare.clicked.connect(self._compare_reports)
        buttons.addWidget(self._compare)
        layout.addLayout(buttons)
        self._status = QLabel("No report yet.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        note = QLabel("Exact IDs must be disjoint. Similarity leakage and exchangeability are "
                      "not checked; class and subgroup coverage are descriptive, not guaranteed. "
                      "This job cannot be stopped or resumed.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _browse(self, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a partition", "",
                                             "Tables (*.csv *.tsv *.txt *.xlsx *.xlsm);;All files (*)")
        if path:
            field.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        self._result.clear()
        self._last_report = None
        self._save.setEnabled(False)
        self._status.setText("Inputs changed; run the evaluation again.")

    def _enable(self, enabled: bool) -> None:
        self._run.setEnabled(enabled)
        self._compare.setEnabled(enabled)
        self._save.setEnabled(enabled and self._last_report is not None)
        for control in self._controls:
            control.setEnabled(enabled)

    def _start(self) -> None:
        from pharmaml.ml.uncertainty_workbench import (
            classification_uncertainty_from_files,
            regression_uncertainty_from_files,
        )

        task = self._task.currentData()
        evaluate = (classification_uncertainty_from_files if task == "classification"
                    else regression_uncertainty_from_files)

        paths = [field.text().strip() for field in self._paths.values()]
        features = [line.strip() for line in self._features.toPlainText().splitlines() if line.strip()]
        target, key = self._target.text().strip(), self._key.text().strip()
        group = self._group.text().strip() or None
        alpha = self._alpha.value()
        diagnostics = self._diagnostics.isChecked()
        if not all(paths) or not features or not target or not key:
            self._result.clear()
            self._status.setText("Select all three files and specify features, target and leakage-unit ID.")
            return
        revision = self._revision
        self._last_report = None
        self._result.clear()
        self._status.setText("Running; fitting and preprocessing use training only.")
        self._enable(False)
        worker = self._runner.submit(
            f"{task.title()} uncertainty",
            lambda _progress: evaluate(
                *(Path(path) for path in paths), features=features, target=target,
                key=key, group=group, alpha=alpha, diagnostics=diagnostics),
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, result: object, revision: int) -> None:
        self._enable(True)
        if revision != self._revision:
            return
        self._last_report = result
        self._save.setEnabled(True)
        self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
        self._status.setText("Evaluation complete. Review assumptions and subgroup sizes below.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Evaluation failed: {error.splitlines()[0]}")

    def _save_report(self) -> None:
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save uncertainty report (new file)", "",
                                             "JSON (*.json);;All files (*)",
                                             options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            with Path(path).open("x", encoding="utf-8") as handle:
                json.dump(self._last_report, handle, indent=2, default=str, allow_nan=False)
                handle.write("\n")
        except (OSError, ValueError, TypeError) as exc:
            self._status.setText(f"Could not save report (existing files are never replaced): {exc}")
            return
        self._status.setText("Report saved. Review for sensitive data before sharing.")

    def _compare_reports(self) -> None:
        first, _ = QFileDialog.getOpenFileName(self, "First uncertainty report", "", "JSON (*.json)")
        if not first:
            return
        second, _ = QFileDialog.getOpenFileName(self, "Second uncertainty report", "", "JSON (*.json)")
        if not second:
            return
        from pharmaml.ml.provenance import compare_uncertainty_reports

        def compare(_progress):
            with Path(first).open(encoding="utf-8") as handle:
                a = json.load(handle)
            with Path(second).open(encoding="utf-8") as handle:
                b = json.load(handle)
            return compare_uncertainty_reports(a, b)

        revision = self._revision
        self._last_report = None
        self._result.clear()
        self._status.setText("Comparing saved reports…")
        self._enable(False)
        worker = self._runner.submit(
            "Compare uncertainty reports", compare,
            on_finished=lambda result: self._comparison_finished(result, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _comparison_finished(self, result: object, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
            self._status.setText("Comparison complete; matching metadata does not prove reproducibility.")

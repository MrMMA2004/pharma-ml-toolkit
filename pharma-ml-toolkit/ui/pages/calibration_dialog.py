"""Independent binary probability calibration report in the shared runner."""
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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class CalibrationDialog(QDialog):
    """Run a fixed binary recalibration experiment; never reuse the current model."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Binary probability recalibration")
        self.resize(760, 690)
        self._runner = runner
        self._revision = 0
        self._last_report: object | None = None
        layout = QVBoxLayout(self)
        intro = QLabel("Select independent training, probability-calibration and test files. "
                       "This fits a new fixed logistic model; it does not reuse the current model.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self._paths: dict[str, QLineEdit] = {}
        self._controls: list[QWidget] = []
        for name, label in (("train", "Training"), ("calibration", "Probability calibration"), ("test", "Test")):
            row = QHBoxLayout()
            field = QLineEdit()
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, edit=field: self._browse(edit))
            field.textChanged.connect(self._invalidate)
            row.addWidget(field, 1)
            row.addWidget(browse)
            form.addRow(label + ":", row)
            self._paths[name] = field
            self._controls.extend((field, browse))
        self._features = QPlainTextEdit()
        self._features.setPlaceholderText("One numeric feature column per line")
        self._features.setMaximumHeight(80)
        self._features.textChanged.connect(self._invalidate)
        form.addRow("Features:", self._features)
        self._controls.append(self._features)
        self._target, self._key, self._positive = QLineEdit(), QLineEdit(), QLineEdit()
        self._positive.setPlaceholderText("Exact text of one of the two classes")
        for title, field in (("Target:", self._target), ("Leakage-unit ID:", self._key),
                             ("Positive label:", self._positive)):
            field.textChanged.connect(self._invalidate)
            form.addRow(title, field)
            self._controls.append(field)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._run = QPushButton("Evaluate recalibration")
        self._run.clicked.connect(self._start)
        self._save = QPushButton("Save report…")
        self._save.setEnabled(False)
        self._save.clicked.connect(self._save_report)
        buttons.addWidget(self._run)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        self._status = QLabel("No report yet. This job cannot be stopped or resumed.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        note = QLabel("Exact IDs must be disjoint, but that does not prove independence. This is "
                      "probability recalibration, not distribution-free interval calibration: a "
                      "later interval/set analysis requires another independent "
                      "calibration set and separate evaluation. Recalibration can worsen probabilities.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _browse(self, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a partition", "",
                                             "Tables (*.csv *.tsv *.txt *.xlsx *.xlsm);;All files (*)")
        if path:
            field.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        if hasattr(self, "_result"):
            self._result.clear()
            self._last_report = None
            self._save.setEnabled(False)
            self._status.setText("Inputs changed; run evaluation again.")

    def _enable(self, enabled: bool) -> None:
        self._run.setEnabled(enabled)
        self._save.setEnabled(enabled and self._last_report is not None)
        for control in self._controls:
            control.setEnabled(enabled)

    def _start(self) -> None:
        from pharmaml.ml.probability_calibration import binary_probability_calibration_from_files

        paths = [field.text().strip() for field in self._paths.values()]
        features = [line.strip() for line in self._features.toPlainText().splitlines() if line.strip()]
        target, key, positive = self._target.text().strip(), self._key.text().strip(), self._positive.text()
        if not all(paths) or not features or not target or not key or not positive.strip():
            self._result.clear()
            self._last_report = None
            self._save.setEnabled(False)
            self._status.setText("Select all three files and specify features, target, ID and positive label.")
            return
        revision = self._revision
        self._last_report = None
        self._result.clear()
        self._status.setText("Running; training and probability calibration use separate files.")
        self._enable(False)
        worker = self._runner.submit(
            "Binary probability recalibration",
            lambda _progress: binary_probability_calibration_from_files(
                *(Path(path) for path in paths), features=features, target=target, key=key, positive_label=positive),
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, result: object, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._last_report = result
            self._save.setEnabled(True)
            self._result.setPlainText(json.dumps(result, indent=2, allow_nan=False))
            self._status.setText("Evaluation complete. Review assumptions and class counts below.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Evaluation failed: {error.splitlines()[0]}")

    def _save_report(self) -> None:
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save recalibration report (new file)", "",
                                             "JSON (*.json);;All files (*)",
                                             options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            with Path(path).open("x", encoding="utf-8") as handle:
                json.dump(self._last_report, handle, indent=2, allow_nan=False)
                handle.write("\n")
        except (OSError, ValueError, TypeError) as exc:
            self._status.setText(f"Could not save report (existing files are never replaced): {exc}")
            return
        self._status.setText("Report saved. Review for sensitive data before sharing.")

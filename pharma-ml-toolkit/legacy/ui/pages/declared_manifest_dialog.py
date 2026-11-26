"""Create a user-declared experiment manifest, without running a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


class DeclaredManifestDialog(QDialog):
    """Fingerprint a chosen file and record declarations for later comparison."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create declared experiment manifest")
        self.resize(760, 620)
        self._runner = runner
        self._revision = 0
        self._report: dict[str, Any] | None = None
        layout = QVBoxLayout(self)
        intro = QLabel("Describe a setup and fingerprint a file. This does not train a model "
                       "or establish that the declared setup was used.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        row = QHBoxLayout()
        self._data = QLineEdit()
        self._data.textChanged.connect(self._invalidate)
        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._browse_data)
        row.addWidget(self._data, 1)
        row.addWidget(self._browse)
        form.addRow("Dataset file:", row)
        self._split = QLineEdit()
        self._split.setPlaceholderText("User-declared split strategy, e.g. group-disjoint")
        self._split.textChanged.connect(self._invalidate)
        form.addRow("Split strategy:", self._split)
        self._parameters = QPlainTextEdit()
        self._parameters.setPlaceholderText("Optional NAME=VALUE entries, one per line")
        self._parameters.setMaximumHeight(100)
        self._parameters.textChanged.connect(self._invalidate)
        form.addRow("Parameters:", self._parameters)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._create = QPushButton("Create manifest")
        self._create.clicked.connect(self._start)
        self._save = QPushButton("Save manifest…")
        self._save.clicked.connect(self._save_manifest)
        self._save.setEnabled(False)
        buttons.addWidget(self._create)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        self._status = QLabel("No manifest yet. File hashing cannot be stopped or resumed.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Unsigned user declarations are not observed run records or proof of "
                         "reproducibility. Keep the input stable while hashing; review file names, "
                         "parameters and environment information for sensitive data before sharing.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self._controls = (self._data, self._browse, self._split, self._parameters, self._create)

    def _browse_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset", "", "All files (*)")
        if path:
            self._data.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        self._report = None
        if hasattr(self, "_result"):
            self._result.clear()
            self._save.setEnabled(False)
            self._status.setText("Inputs changed; create the manifest again.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)
        self._save.setEnabled(enabled and self._report is not None)

    def _start(self) -> None:
        from pharmaml.ml.provenance import experiment_manifest

        data = self._data.text().strip()
        split = self._split.text().strip()
        parameters: dict[str, str] = {}
        for line in self._parameters.toPlainText().splitlines():
            if not line.strip():
                continue
            name, sep, value = line.partition("=")
            name = name.strip()
            if not sep or not name or name in parameters:
                self._invalidate()
                self._status.setText("Parameters must be unique, nonempty NAME=VALUE entries.")
                return
            parameters[name] = value
        if not data or not split:
            self._invalidate()
            self._status.setText("Select a dataset and enter a split strategy.")
            return
        revision = self._revision
        self._report = None
        self._result.clear()
        self._status.setText("Fingerprinting file; this job cannot be stopped or resumed.")
        self._enable(False)
        worker = self._runner.submit(
            "Declared experiment manifest",
            lambda _progress: experiment_manifest(Path(data), parameters=parameters, split_strategy=split),
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, report: dict[str, Any], revision: int) -> None:
        self._enable(True)
        if revision != self._revision:
            return
        self._report = report
        self._result.setPlainText(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        self._save.setEnabled(True)
        self._status.setText("Manifest created from user declarations; no training was observed.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Manifest failed: {error.splitlines()[0]}")

    def _save_manifest(self) -> None:
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save manifest (new JSON file)", "", "JSON (*.json)",
                                             options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            if Path(path).suffix.lower() != ".json":
                raise ValueError("Choose a .json file")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._report, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        except (OSError, ValueError, TypeError) as exc:
            self._status.setText(f"Could not save manifest; existing files are never replaced: {exc}")
            return
        self._status.setText("Manifest saved. Review it for sensitive information before sharing.")

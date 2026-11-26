"""Read-only environment, strict example-data and checkpoint diagnostics."""
from __future__ import annotations

import json
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

_TOOLKIT = Path(__file__).resolve().parents[3]


class ReproducibilityDialog(QDialog):
    """Run existing read-only checks without silently changing any artifacts."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reproducibility and checkpoint diagnostics")
        self.resize(760, 650)
        self._runner = runner
        self._revision = 0
        layout = QVBoxLayout(self)
        description = QLabel("Read-only diagnostics and optional declared-manifest export. No job is resumed, "
                             "input data is not modified, and numeric drift is not treated as a byte match.")
        description.setWordWrap(True)
        layout.addWidget(description)
        buttons = QHBoxLayout()
        self._environment = QPushButton("Environment")
        self._environment.clicked.connect(self._show_environment)
        self._drift = QPushButton("Strict example-data drift")
        self._drift.clicked.connect(self._show_drift)
        buttons.addWidget(self._environment)
        buttons.addWidget(self._drift)
        layout.addLayout(buttons)
        self._create_manifest = QPushButton("Create declared manifest…")
        self._create_manifest.clicked.connect(self._show_create_manifest)
        layout.addWidget(self._create_manifest)
        comparisons = QHBoxLayout()
        self._compare_training = QPushButton("Compare training records…")
        self._compare_training.clicked.connect(self._show_training_comparison)
        comparisons.addWidget(self._compare_training)
        self._compare_manifests = QPushButton("Compare declared manifests…")
        self._compare_manifests.clicked.connect(self._show_manifest_comparison)
        comparisons.addWidget(self._compare_manifests)
        layout.addLayout(comparisons)
        note = QLabel("Example-data drift requires checkout tools and data; the check runs in a child "
                      "process and may take time. These jobs cannot be stopped or resumed.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        row = QHBoxLayout()
        self._checkpoint = QLineEdit()
        self._checkpoint.setToolTip("Path to the training checkpoint to restore")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._checkpoint.textChanged.connect(self._invalidate)
        row.addWidget(self._checkpoint, 1)
        row.addWidget(browse)
        form.addRow("Checkpoint JSON:", row)
        self._inputs = QPlainTextEdit()
        self._inputs.setMaximumHeight(70)
        self._inputs.setPlaceholderText("Optional complete set of recorded input paths, one per line")
        self._inputs.textChanged.connect(self._invalidate)
        form.addRow("Recorded inputs:", self._inputs)
        self._inspect = QPushButton("Inspect checkpoint")
        self._inspect.clicked.connect(self._show_checkpoint)
        form.addRow("", self._inspect)
        layout.addLayout(form)
        self._controls: tuple[QWidget, ...] = (self._environment, self._drift, self._create_manifest,
                                               self._compare_training, self._compare_manifests, self._checkpoint,
                                               browse, self._inputs, self._inspect)
        self._status = QLabel("Choose a diagnostic; no checks have run.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Results may contain sensitive paths and environment information. "
                         "Matching fingerprints do not guarantee safe resume or reproducibility.")
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", "", "JSON (*.json);;All files (*)")
        if path:
            self._checkpoint.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        if hasattr(self, "_result"):
            self._result.clear()
            self._status.setText("Inputs changed; rerun the desired diagnostic.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)

    def _submit(self, name: str, job) -> None:
        revision = self._revision
        self._result.clear()
        self._status.setText(f"Running {name}; no data or checkpoints are modified.")
        self._enable(False)
        worker = self._runner.submit(name, lambda _progress: job(),
                                     on_finished=lambda result: self._finished(name, result, revision),
                                     on_failed=lambda error: self._failed(error, revision))
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _show_environment(self) -> None:
        from pharmaml.ml.provenance import environment_snapshot

        self._submit("Environment", environment_snapshot)

    def _show_drift(self) -> None:
        from pharmaml.ml.studio_diagnostics import run_example_drift_diagnostic

        script = _TOOLKIT / "tools" / "diagnose_data_drift.py"
        self._submit("Strict example-data drift", lambda: run_example_drift_diagnostic(script))

    def _show_training_comparison(self) -> None:
        first, _ = QFileDialog.getOpenFileName(self, "First training run record", "", "JSON (*.json)")
        if not first:
            return
        second, _ = QFileDialog.getOpenFileName(self, "Second training run record", "", "JSON (*.json)")
        if not second:
            return
        from pharmaml.ml.training_provenance import compare_training_records

        def compare():
            with Path(first).open(encoding="utf-8") as handle:
                left = json.load(handle)
            with Path(second).open(encoding="utf-8") as handle:
                right = json.load(handle)
            return compare_training_records(left, right)

        self._submit("Training record comparison", compare)

    def _show_create_manifest(self) -> None:
        from .declared_manifest_dialog import DeclaredManifestDialog

        dialog = getattr(self, "_manifest_dialog", None)
        if dialog is None:
            dialog = DeclaredManifestDialog(self._runner, self)
            self._manifest_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _show_manifest_comparison(self) -> None:
        first, _ = QFileDialog.getOpenFileName(self, "First declared experiment manifest", "", "JSON (*.json)")
        if not first:
            return
        second, _ = QFileDialog.getOpenFileName(self, "Second declared experiment manifest", "", "JSON (*.json)")
        if not second:
            return
        from pharmaml.ml.provenance import compare_manifests

        def compare():
            with Path(first).open(encoding="utf-8") as handle:
                left = json.load(handle)
            with Path(second).open(encoding="utf-8") as handle:
                right = json.load(handle)
            return compare_manifests(left, right)

        self._submit("Declared manifest comparison", compare)

    def _show_checkpoint(self) -> None:
        from pharmaml.ml.checkpoint_diagnostics import inspect_checkpoint

        path = self._checkpoint.text().strip()
        inputs = [line.strip() for line in self._inputs.toPlainText().splitlines() if line.strip()]
        if not path:
            self._result.clear()
            self._status.setText("Select a checkpoint JSON first.")
            return
        self._submit("Checkpoint inspection", lambda: inspect_checkpoint(path, inputs=inputs))

    def _finished(self, name: str, result: Any, revision: int) -> None:
        self._enable(True)
        if revision != self._revision:
            return
        self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
        if name == "Strict example-data drift":
            status = "STRICT BYTE DRIFT DETECTED" if result["strict_check_exit_code"] else "strict byte check passed"
            self._status.setText(f"{status}. Numeric descriptions never waive a strict mismatch.")
        elif name == "Checkpoint inspection" and (
            not result["valid_envelope"] or result["input_status"].startswith("changed")
            or result["implementation_status"].startswith("changed")
        ):
            self._status.setText("Checkpoint warning: review the result; no resume was attempted.")
        else:
            self._status.setText("Diagnostic complete. Matching results do not prove reproducibility or safe resume.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Diagnostic failed: {error.splitlines()[0]}")

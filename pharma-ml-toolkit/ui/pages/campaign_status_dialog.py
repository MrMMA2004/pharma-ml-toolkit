"""Read-only inspection of a reviewed Vina campaign output."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
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


class CampaignStatusDialog(QDialog):
    """Inspect recorded progress and hashes without offering resume."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inspect Vina campaign progress")
        self.resize(760, 590)
        self._runner = runner
        self._revision = 0
        layout = QVBoxLayout(self)
        note = QLabel("Read-only inspection of a review-first Vina campaign directory. "
                      "Stop other writers before inspection; a running process cannot be detected.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        row = QHBoxLayout()
        self._path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._path.textChanged.connect(self._invalidate)
        row.addWidget(self._path, 1)
        row.addWidget(browse)
        form.addRow("Campaign directory:", row)
        self._originals = QCheckBox("Also check original input and engine files (reads recorded external paths)")
        self._originals.stateChanged.connect(self._invalidate)
        form.addRow("", self._originals)
        self._inspect = QPushButton("Inspect campaign")
        self._inspect.clicked.connect(self._start)
        form.addRow("", self._inspect)
        layout.addLayout(form)
        self._controls: tuple[QWidget, ...] = (self._path, browse, self._originals, self._inspect)
        self._status = QLabel("No inspection yet.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Unsigned manifest; matching hashes do not validate a docking result or authorize resume. "
                         "Unrecorded files are never treated as complete. No files are changed and no resume is available.")
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select campaign output directory")
        if path:
            self._path.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        if hasattr(self, "_result"):
            self._result.clear()
            self._status.setText("Inputs changed; inspect again.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)

    def _start(self) -> None:
        from pharmaml.docking.campaign_status import inspect_vina_campaign

        path = self._path.text().strip()
        if not path:
            self._status.setText("Select a campaign output directory.")
            self._result.clear()
            return
        originals, revision = self._originals.isChecked(), self._revision
        self._result.clear()
        self._enable(False)
        self._status.setText("Inspecting campaign; no files will be modified.")
        worker = self._runner.submit(
            "Inspect Vina campaign", lambda _progress: inspect_vina_campaign(path, check_originals=originals),
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, result: object, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.setPlainText(json.dumps(result, indent=2, allow_nan=False))
            self._status.setText("Inspection complete; review changed, missing and unrecorded items. No resume is available.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Inspection failed: {error.splitlines()[0]}")

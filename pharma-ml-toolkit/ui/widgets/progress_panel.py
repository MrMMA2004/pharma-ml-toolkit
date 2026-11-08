"""Progress panel: progress bar + cooperative Stop (directive 16.14).

Rides the round-205 lesson: the Stop button is checkpoint-honest -- it
requests a cooperative stop through :class:`JobRunner` and the label
says what is actually happening ("Stopping… (takes effect at the next
checkpoint)"), because backend loops only observe the flag at their
checkpoints.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class ProgressPanel(QWidget):
    """A labelled progress bar with a checkpoint-honest Stop button."""

    stop_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._label = QLabel("Ready")
        self._label.setObjectName("studioProgressLabel")
        self._stop = QPushButton("Stop")
        self._stop.setToolTip("Request a cooperative stop — takes effect at the job's next checkpoint")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._on_stop)
        row.addWidget(self._bar, stretch=1)
        row.addWidget(self._stop)
        layout.addLayout(row)
        layout.addWidget(self._label)
        self._note = QLabel("")
        self._note.setObjectName("studioProgressNote")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self.hide()

    # -- state transitions -----------------------------------------------------
    def begin(self, job_name: str) -> None:
        self._bar.setValue(0)
        self._label.setText(job_name)
        self._note.setText("")
        self._stop.setEnabled(True)
        self.show()

    def update_progress(self, percent: int, label: str) -> None:
        self._bar.setValue(max(0, min(100, int(percent))))
        if label:
            self._label.setText(label)

    def note(self, text: str) -> None:
        self._note.setText(text)

    def end(self) -> None:
        self._bar.setValue(0)
        self._stop.setEnabled(False)
        self._label.setText("Ready")
        self._note.setText("")

    # -- wiring convenience ------------------------------------------------------
    def attach(self, runner: JobRunner) -> None:
        """One-call wiring of runner signals onto this panel."""
        runner.job_started.connect(self.begin)
        runner.job_ended.connect(self.end)
        self.stop_clicked.connect(runner.request_stop)

    def _on_stop(self) -> None:
        self._stop.setEnabled(False)
        self._label.setText("Stopping… (takes effect at the next checkpoint)")
        self.stop_clicked.emit()

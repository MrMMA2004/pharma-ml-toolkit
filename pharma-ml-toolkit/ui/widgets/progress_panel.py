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
        # review M-6 fix: elapsed time from the start of THIS job
        import time as _time
        self._t0 = _time.monotonic()
        from PySide6.QtCore import QTimer
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._show_elapsed)
        self._tick.start(1000)
        self._job_name = job_name
        self._show_elapsed()

    def _show_elapsed(self) -> None:
        elapsed = getattr(self, "_t0", None)
        if elapsed is None:
            return
        import time as _time
        secs = _time.monotonic() - elapsed
        name = getattr(self, "_job_name", "")
        self._label.setText(f"{name} — {int(secs)}s elapsed")

    def update_progress(self, percent: int, label: str) -> None:
        self._bar.setValue(max(0, min(100, int(percent))))
        if label:
            self._job_name = label
            self._show_elapsed()

    def note(self, text: str) -> None:
        self._note.setText(text)

    def end(self) -> None:
        tick = getattr(self, "_tick", None)
        if tick is not None:
            tick.stop()

        self._bar.setValue(0)
        self._stop.setEnabled(False)
        self._label.setText("Ready")
        self._note.setText("")
        # review m-6: begin() shows the panel; a finished job must give
        # the row back — otherwise every page keeps a dead "Ready" bar
        # after its first run. A stopped job keeps the panel visible
        # briefly to show the honest stop note.
        if not self._note.text():
            self.hide()

    # -- wiring convenience ------------------------------------------------------
    def attach(self, runner: JobRunner) -> None:
        """One-call wiring of runner signals onto this panel."""
        runner.job_started.connect(self.begin)
        runner.job_ended.connect(self.end)
        # review 4.7: the relayed progress (pages that never passed
        # on_progress still move this bar) and the honest stop line
        # (a cooperative Stop no longer leaves a stale "Ready").
        runner.job_progress.connect(self.update_progress)
        runner.job_stopped.connect(self._on_stopped)
        self.stop_clicked.connect(runner.request_stop)

    def _on_stopped(self, stopped_after: float) -> None:
        """A cooperative stop reached its checkpoint (review 4.7)."""
        self._stop.setEnabled(False)
        self._job_name = ""
        self._note.setText(
            f"Stopped after {stopped_after:.1f}s (partial results, if the "
            "job preserved any, appear in the results area)"
        )

    def _on_stop(self) -> None:
        self._stop.setEnabled(False)
        self._label.setText("Stopping… (takes effect at the next checkpoint)")
        self.stop_clicked.emit()

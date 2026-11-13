"""The unified job center (part6 §10, Phase 4).

One dialog for the Studio's computational jobs:

* the LIVE job — name, elapsed runtime (a monotonic clock, not wall
  guesses), the honest cooperative Stop (``runner.request_stop``; the
  §13 rules forbid faking Pause/Resume without a tested checkpoint,
  so no Pause button exists here), and the runner's real busy state;
* the HISTORY — the recorded runs from ``pharmaml.studio.runs`` (the
  same files the pages' ``record_run`` calls have been writing since
  v1.2.171), newest first, with the §9 provenance detail (toolkit
  version, created timestamp, the run's own summary block) on select.

Honesty rules baked in: no job is ever shown as "queued" (the runner
runs exactly one job at a time and this dialog says so); a failed
run-record is reported by its absence, never by a guessed payload
(the backend's own contract); the runtime display ticks only while a
job is actually running.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

#: How the elapsed clock renders.
def _fmt_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:d}:{secs:02d}"


class JobCenterDialog(QDialog):
    """Live job + honest run history, over the shared JobRunner."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Job center")
        self.resize(760, 480)
        self._runner = runner
        self._started_at: float | None = None

        lay = QVBoxLayout(self)

        # -- the live job card ------------------------------------------------
        live = QHBoxLayout()
        self._live_label = QLabel("No job running.")
        self._live_label.setObjectName("studioStatus")
        self._live_label.setWordWrap(True)
        live.addWidget(self._live_label, stretch=1)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip(
            "Cooperative Stop: asks the running job to cancel at its "
            "next checkpoint. There is deliberately no Pause/Resume — "
            "that needs a per-workflow tested checkpoint contract."
        )
        self._stop_btn.clicked.connect(runner.request_stop)
        self._stop_btn.setEnabled(False)
        live.addWidget(self._stop_btn)
        lay.addLayout(live)

        # -- the history table --------------------------------------------------
        lay.addWidget(QLabel("Recorded run history (provenance on select):"))
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._history = QListWidget()
        self._history.currentItemChanged.connect(self._show_detail)
        splitter.addWidget(self._history)
        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText(
            "Select a run to see its provenance: toolkit version, "
            "timestamp and the run's own recorded summary."
        )
        splitter.addWidget(self._detail)
        splitter.setSizes([260, 200])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)

        # the runtime ticker: only ticks while a job is really running
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)

        self._refresh_history()
        self._sync_live()

    # -- live job ------------------------------------------------------------------
    def _on_started(self, name: str) -> None:
        self._started_at = time.monotonic()
        self._ticker.start()
        self._live_label.setText(f"Running: {name} · 0:00")
        self._stop_btn.setEnabled(True)

    def _on_ended(self) -> None:
        self._started_at = None
        self._ticker.stop()
        self._sync_live()
        self._refresh_history()  # the finished job may have recorded a run

    def _tick(self) -> None:
        if self._started_at is not None:
            elapsed = time.monotonic() - self._started_at
            name = self._runner.job_name or "job"
            self._live_label.setText(f"Running: {name} · {_fmt_elapsed(elapsed)}")

    def _sync_live(self) -> None:
        busy = self._runner.busy
        self._stop_btn.setEnabled(busy)
        if not busy:
            self._live_label.setText(
                "No job running. The Studio runs one computational job "
                "at a time; its record appears below when it finishes."
            )
        elif self._started_at is None:
            # busy but started before this dialog opened: count from now
            # and say so — never guess an earlier start
            self._started_at = time.monotonic()
            self._ticker.start()
            self._live_label.setText(
                f"Running: {self._runner.job_name} · (opened mid-run; "
                "clock counts from the dialog opening)"
            )

    # -- history -------------------------------------------------------------------
    def _refresh_history(self) -> None:
        from pharmaml.studio.runs import latest_runs, run_families

        self._history.clear()
        families = run_families()
        total = 0
        for family, count in sorted(families.items()):
            total += count
            for record in latest_runs(family, limit=10):
                created = str(record.get("created", ""))[:19].replace("T", " ")
                label = str(record.get("label", "(unlabelled)"))
                item = QListWidgetItem(f"{created}  [{family}]  {label}")
                item.setData(Qt.ItemDataRole.UserRole, record)
                self._history.addItem(item)
        if self._history.count():
            self._history.setCurrentRow(0)
        else:
            self._detail.setPlainText(
                "No recorded runs yet. Runs are recorded by the workflows "
                "that produce real results (benchmarks, training, docking "
                "campaigns, exports…); a failed or cancelled run records "
                "only what its workflow chose to record."
            )

    def _show_detail(self, current: QListWidgetItem | None,
                     _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        record: dict[str, Any] = current.data(Qt.ItemDataRole.UserRole)
        lines = [
            f"family: {record.get('family', '?')}",
            f"label:  {record.get('label', '?')}",
            f"created: {record.get('created', '?')}",
            f"toolkit version: {record.get('toolkit_version', '?')}",
            "recorded summary:",
        ]
        summary = record.get("summary") or {}
        for key in sorted(summary):
            lines.append(f"  {key}: {summary[key]}")
        lines.append(
            "(fields beyond this block were not recorded by the workflow; "
            "nothing here is inferred — see the run's own page for the "
            "full context)"
        )
        self._detail.setPlainText("\n".join(lines))

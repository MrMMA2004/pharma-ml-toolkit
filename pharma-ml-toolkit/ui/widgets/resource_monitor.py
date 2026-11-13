"""§17E — the live resource-monitor widget.

A small always-true card for the Job Center: what the system can
actually evidence (CPU threads, memory with its source), the active
execution preset (with its honest trade-off sentence, switchable),
and the currently running job driven by the shared JobRunner's own
signals — never a guess.

Honesty rules (they are the whole point of §17E):

* memory that could not be evidenced renders ``unknown`` with the
  source line saying why — never a fabricated number;
* there is deliberately NO GPU row when no accelerator backend is
  importable: the widget shows an explicit "not detected" line only
  in the details, because a permanently empty row would imply the
  toolkit claims GPU awareness it does not have;
* the refresh QTimer runs ONLY while the widget is visible
  (showEvent/hideEvent) — no background polling tax on the very
  resources it reports.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pharmaml.utils.resources import (
    PRESETS,
    find_preset,
    snapshot,
    store_preset,
    stored_preset,
)

_REFRESH_MS = 2000


def _gpu_line() -> str:
    """The honest accelerator status: importable or not, that's it."""
    try:  # pragma: no cover - depends on the machine's CUDA install
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            return f"GPU: {count} CUDA device(s) (torch)"
    except Exception:  # noqa: BLE001 - an unimportable backend is a fact
        pass
    return "GPU: none detected (CPU-only build) — shown as fact, " \
           "not as a failure"


class ResourceMonitorWidget(QWidget):
    """Live CPU/memory/preset/job card over the shared JobRunner."""

    def __init__(self, runner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runner = runner

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self._cpu = QLabel("CPU threads: …")
        self._mem = QLabel("Memory available: …")
        self._mem.setToolTip(
            "The smaller of the cgroup limit and /proc/meminfo MemAvailable "
            "— with the source named. 'unknown' means neither could be "
            "read; a number is never invented.")
        top.addWidget(self._cpu)
        top.addWidget(self._mem, stretch=1)
        lay.addLayout(top)

        mid = QHBoxLayout()
        mid.addWidget(QLabel("Execution preset:"))
        self._preset = QComboBox()
        for preset in PRESETS:
            self._preset.addItem(preset.label, preset.key)
        current = stored_preset()
        index = self._preset.findData(current)
        if index >= 0:
            self._preset.setCurrentIndex(index)
        self._preset.currentIndexChanged.connect(self._on_preset)
        mid.addWidget(self._preset)
        self._trades = QLabel("")
        self._trades.setObjectName("studioStatus")
        self._trades.setWordWrap(True)
        mid.addWidget(self._trades, stretch=1)
        lay.addLayout(mid)

        self._job = QLabel("No job running.")
        self._job.setObjectName("studioStatus")
        lay.addWidget(self._job)
        self._gpu_note = QLabel("")
        self._gpu_note.setWordWrap(True)
        lay.addWidget(self._gpu_note)

        # the timer runs only while visible (§17E discipline)
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

        # the active-job row rides the runner's own signals
        if runner is not None:
            runner.job_started.connect(self._on_started)
            runner.job_ended.connect(self._on_ended)

        self.refresh()

    # -- live snapshot ------------------------------------------------------

    def refresh(self) -> None:
        snap = snapshot()
        self._cpu.setText(f"CPU threads: {snap.cpu_count}")
        self._mem.setText(
            f"Memory available: {snap.memory_gib()}"
            f" (source: {snap.memory_source})")
        preset = find_preset(stored_preset())
        self._trades.setText(preset.trades)
        self._gpu_note.setText(_gpu_line())

    # -- preset switching ---------------------------------------------------

    def _on_preset(self, index: int) -> None:
        key = self._preset.itemData(index)
        if key and key != stored_preset():
            if store_preset(key):
                self.refresh()

    # -- active job ---------------------------------------------------------

    def _on_started(self, name: str) -> None:
        self._job.setText(f"Running: {name}")

    def _on_ended(self) -> None:
        self._job.setText("No job running.")

    # -- timer discipline ---------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._timer.stop()
        super().hideEvent(event)

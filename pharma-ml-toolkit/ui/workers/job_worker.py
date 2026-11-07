"""Background job execution for PharmaML Studio (directive 16.14).

Long-running backend work NEVER runs on the GUI thread: ``JobWorker``
is a ``QRunnable`` pushed onto the global ``QThreadPool`` whose signals
(progress / finished / failed / cancelled) marshal back to the GUI
thread via Qt's queued connections.

Cooperative stop: the worker hands the backend callable a progress
callback. Backend checkpoint loops that follow the toolkit convention
(round 205's ``JobCancelled`` channel) treat a ``False`` return as
"stop requested" and raise ``pharmaml.exceptions.JobCancelled``; the
worker catches it and reports a clean cancellation -- never a red
traceback. Callables that ignore the callback's return value simply
run to completion (documented honestly in the GUI docs).

Duplicate-submission guard: :class:`JobRunner` refuses a second job
while one is running (the "prevent duplicate submissions" clause).
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

ProgressReport = Callable[[Any, str], None]
ProgressCallback = Callable[[Any, str], bool]
BackendJob = Callable[[ProgressCallback], Any]


class JobSignals(QObject):
    """Queued-connection bridge from the worker thread to the GUI thread."""

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(float)


class JobWorker(QRunnable):
    """Run one backend callable off the GUI thread."""

    def __init__(self, job: BackendJob) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self._stop = False
        self._t0 = time.monotonic()
        self.setAutoDelete(True)

    # -- control ------------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the running job to stop at its next checkpoint."""
        self._stop = True

    # -- the progress callback handed to the backend -------------------------
    def _progress(self, fraction: Any, label: str = "") -> bool:
        """Report progress; the RETURN tells the backend to keep going.

        Returning ``False`` is the toolkit's cooperative-stop signal:
        convention-following backend loops raise ``JobCancelled``.
        """
        try:
            percent = round(float(fraction) * 100)
        except (TypeError, ValueError):
            percent = 0
        self.signals.progress.emit(percent, label)
        return not self._stop

    # -- QRunnable entry point ------------------------------------------------
    def run(self) -> None:
        try:
            result = self.job(self._progress)
        except BaseException as exc:  # noqa: BLE001 - marshalled, not swallowed
            from pharmaml.exceptions import JobCancelled

            if isinstance(exc, JobCancelled):
                self.signals.cancelled.emit(time.monotonic() - self._t0)
                return
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
            return
        self.signals.finished.emit(result)


class JobRunner(QObject):
    """Owns the single in-flight job; pages drive their buttons through it."""

    job_started = Signal(str)
    job_ended = Signal()

    def __init__(self, parent: QObject | None = None, pool: QThreadPool | None = None) -> None:
        super().__init__(parent)
        self._pool = pool or QThreadPool.globalInstance()
        self._worker: JobWorker | None = None
        self._busy = False
        self._job_name = ""

    # -- queries -------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def job_name(self) -> str:
        return self._job_name

    # -- control --------------------------------------------------------------
    def submit(
        self,
        name: str,
        job: BackendJob,
        on_finished: Callable[[Any], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
        on_cancelled: Callable[[float], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> JobWorker | None:
        """Start ``job`` unless one is already running (returns the worker).

        The optional ``on_*`` handlers are connected BEFORE the runnable
        enters the pool (round 208): ``pool.start`` hands the worker to
        another thread immediately, so a page that connects
        ``worker.signals.finished`` *after* ``submit`` returns races the
        emitting thread and can lose the result of an instant job —
        the UI stays on "Running …" forever. Wiring the handlers inside
        ``submit`` closes that window. The runner's own finishers stay
        the first connections, so ``job_ended`` still fires before the
        page's result slot (the ordering the offscreen tests settle on).
        """
        if self._busy:
            return None
        self._busy = True
        self._job_name = name
        worker = JobWorker(job)
        self._worker = worker
        self.job_started.emit(name)
        worker.signals.finished.connect(lambda _r: self._finish())
        worker.signals.failed.connect(lambda _m: self._finish())
        worker.signals.cancelled.connect(lambda _s: self._finish())
        if on_finished is not None:
            worker.signals.finished.connect(on_finished)
        if on_failed is not None:
            worker.signals.failed.connect(on_failed)
        if on_cancelled is not None:
            worker.signals.cancelled.connect(on_cancelled)
        if on_progress is not None:
            worker.signals.progress.connect(on_progress)
        self._pool.start(worker)
        return worker

    def request_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    # -- internals --------------------------------------------------------------
    def _finish(self) -> None:
        self._busy = False
        self._job_name = ""
        self._worker = None
        self.job_ended.emit()

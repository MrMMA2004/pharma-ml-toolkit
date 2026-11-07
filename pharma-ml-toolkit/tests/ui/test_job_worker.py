"""Job worker tests (16.14): off-thread execution, progress, cooperative stop."""

from __future__ import annotations

import time

from pharmaml.exceptions import JobCancelled


def _drain(pool_wait: float = 5.0) -> None:
    """Wait for the global thread pool to finish queued workers."""
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def test_finished_carries_real_result(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    results: list[object] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        progress(0.5, "half")
        return 6 * 7

    worker = runner.submit("answer", job)
    assert worker is not None
    worker.signals.finished.connect(results.append)
    done.wait()
    _drain()
    assert results == [42]


def test_failed_marshals_error(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    errors: list[str] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        raise ZeroDivisionError("integer division by zero")

    worker = runner.submit("boom", job)
    worker.signals.failed.connect(errors.append)
    done.wait()
    _drain()
    assert errors and "ZeroDivisionError" in errors[0]


def test_cooperative_stop_lands_as_cancelled(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    cancelled: list[float] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        for i in range(1, 101):
            keep_going = progress(i / 100, f"checkpoint {i}/100")
            if not keep_going:
                raise JobCancelled("user stop")
            time.sleep(0.005)
        return "completed"

    worker = runner.submit("long", job)
    worker.signals.cancelled.connect(cancelled.append)
    worker.request_stop()  # observed at the next checkpoint
    done.wait()
    _drain()
    assert cancelled and cancelled[0] >= 0.0  # seconds elapsed, cleanly cancelled


def test_duplicate_submission_refused(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        time.sleep(0.15)

    first = runner.submit("one", job)
    assert first is not None
    second = runner.submit("two", job)  # refused while busy (16.14)
    assert second is None
    done.wait()
    _drain()
    assert not runner.busy


def test_progress_signal_reaches_gui_thread(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    seen: list[int] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        for i in range(1, 5):
            progress(i / 4, "step")

    worker = runner.submit("progress", job)
    worker.signals.progress.connect(lambda pct, label: seen.append(pct))
    done.wait()
    _drain()
    assert seen == [25, 50, 75, 100]

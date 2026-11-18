"""Job worker tests (16.14): off-thread execution, progress, cooperative stop."""

from __future__ import annotations

import time

from pharmaml.exceptions import JobCancelled


def _drain(pool_wait: float = 5.0) -> None:
    """Wait for the global thread pool to finish queued workers.

    Review 4.7 hardening: also pump the GUI event queue once the pool is
    idle. Cross-thread signal emissions (cancelled, then preserved) are
    QUEUED events: a ``waitSignal(job_ended)`` nested loop returns the
    moment job_ended is delivered — which happens inside the cancelled
    delivery, potentially BEFORE the preserved event right behind it in
    the queue has run. Pumping after the pool drains closes that window
    without changing any emission order (probed: the clean pre-4.7
    runner showed the identical 0-delivered-preserved state; timing
    alone decided whether the old test caught it).
    """
    from PySide6.QtCore import QCoreApplication, QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    # the queue behind the last delivered signal
    for _ in range(4):
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_finished_carries_real_result(studio_app, qtbot) -> None:
    """The result object travels through ``finished`` to the GUI thread.

    Wired via ``submit(on_finished=...)`` (round 208): the handler is
    connected before ``pool.start``, so an instant job can never race
    its own slot (the connect-after-submit form of this test lost the
    emission under load and failed with an empty list).
    """
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    results: list[object] = []

    def job(progress):
        progress(0.5, "half")
        return 6 * 7

    worker = runner.submit("answer", job, on_finished=results.append)
    assert worker is not None
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)
    done.wait()
    qtbot.wait(50)
    assert results == [42]


def test_failed_marshals_error(studio_app, qtbot) -> None:
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    errors: list[str] = []

    def job(progress):
        raise ZeroDivisionError("integer division by zero")

    worker = runner.submit("boom", job, on_failed=errors.append)
    assert worker is not None
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)
    done.wait()
    qtbot.wait(50)
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

    worker = runner.submit("long", job, on_cancelled=cancelled.append)
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

    def job(progress):
        for i in range(1, 5):
            progress(i / 4, "step")

    def on_progress(pct: int, label: str) -> None:
        seen.append(pct)

    worker = runner.submit("progress", job, on_progress=on_progress)
    assert worker is not None
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)
    done.wait()
    qtbot.wait(50)
    assert seen == [25, 50, 75, 100]


def test_submit_handlers_connected_before_start(studio_app, qtbot) -> None:
    """Round 208 lost-update race: handlers are wired BEFORE pool.start.

    ``pool.start`` hands the runnable to another thread immediately; a
    caller that connects ``worker.signals.finished`` after ``submit``
    returns races the emitting thread and can lose the result of an
    instant job (the page stays on "Running …" forever — the flake the
    pkpd offscreen suite caught). Handlers passed to ``submit`` are
    connected before the start, and the runner's own finisher stays the
    first connection, so ``job_ended`` lands before the result slot.
    """
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    landed: list[str] = []
    runner.job_ended.connect(lambda: landed.append("ended"))

    def make_job(tag: str):
        def job(progress):
            return f"result-{tag}"

        return job

    def on_finished(result: object) -> None:
        landed.append(str(result))

    for i in range(12):
        tag = f"tag{i}"
        worker = runner.submit(f"instant {i}", make_job(tag), on_finished=on_finished)
        assert worker is not None
        _drain()
        qtbot.wait(50)  # one queued-slot pump per the round-207 lesson
        assert landed == ["ended", f"result-{tag}"], landed
        landed.clear()
        assert not runner.busy


def test_submit_failure_handler_before_start(studio_app, qtbot) -> None:
    """The on_failed handler is wired before the start too (race closure)."""
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    errors: list[str] = []

    def job(progress):
        raise RuntimeError("boom-208")

    worker = runner.submit("boom", job, on_failed=errors.append)
    assert worker is not None
    _drain()
    qtbot.wait(50)
    assert errors and "RuntimeError" in errors[0]
    assert not runner.busy


def test_finished_slots_survive_the_worker_gc_race(studio_app, qtbot) -> None:
    """Regression (round 227): late slots must outlive the finisher.

    The runner's ``_finish`` is the FIRST slot of the finished emission;
    dropping the last worker reference there used to let CPython's
    immediate refcount destruction tear down the worker's JobSignals
    mid-delivery, silently swallowing every page slot connected after
    it.  The Utilities page's slow-import campaign jobs hit this
    deterministically (the report stayed on "Running …" while
    ``busy`` already flipped false).  The retired-workers hold fixes
    it: a forced ``gc.collect()`` inside the first page slot must not
    be able to kill the late ones either.
    """
    import gc
    import time as _time

    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    order: list[str] = []
    runner.job_ended.connect(lambda: order.append("job_ended"))

    def job(progress):
        _time.sleep(0.05)  # land the emit while the main thread pumps events
        return "payload"

    def first_slot(result: object) -> None:
        order.append("first")
        gc.collect()  # the round-227 trigger, made deterministic

    def late_slot(result: object) -> None:
        order.append("late")

    worker = runner.submit("gc probe", job, on_finished=first_slot)
    assert worker is not None
    worker.signals.finished.connect(late_slot)  # connected AFTER submit, page-style
    deadline = _time.monotonic() + 10.0
    while _time.monotonic() < deadline and "late" not in order:
        studio_app.processEvents()
        _time.sleep(0.01)
    assert order == ["job_ended", "first", "late"], order
    assert not runner.busy


# ---------------------------------------------------------------------------
# (part5-continuation round 12, §12/§13) the preserved-partial channel
# ---------------------------------------------------------------------------
def test_cancelled_partial_rides_the_preserved_signal(
    studio_app, qtbot
) -> None:
    """A JobCancelled with partial_results emits preserved right after
    cancelled: the six registered §13 operations attach their payload
    on a cooperative stop, and the worker must hand it to the GUI
    thread instead of dropping it on the floor."""
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    cancelled: list[float] = []
    preserved: list[object] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    payload = {"kept": 12, "of": 40}  # any object rides the channel

    def job(progress):
        for i in range(1, 101):
            if not progress(i / 100, f"checkpoint {i}/100"):
                raise JobCancelled("user stop", partial_results=payload)
            time.sleep(0.005)
        return "completed"

    worker = runner.submit(
        "partial", job,
        on_cancelled=cancelled.append,
        on_preserved=preserved.append,
    )
    assert worker is not None
    worker.request_stop()
    done.wait()
    _drain()
    assert cancelled  # the historical float payload still fires
    assert preserved == [payload]  # and the partial rode through


def test_cancelled_without_partial_stays_silent(
    studio_app, qtbot
) -> None:
    """A JobCancelled with NO payload (the historical convention, the
    five non-§13 backends) emits cancelled only -- on_preserved
    handlers never fire, so pages opt in without noise."""
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    cancelled: list[float] = []
    preserved: list[object] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        for i in range(1, 101):
            if not progress(i / 100, f"checkpoint {i}/100"):
                raise JobCancelled("user stop")  # no partial_results
            time.sleep(0.005)
        return "completed"

    worker = runner.submit(
        "silent", job,
        on_cancelled=cancelled.append,
        on_preserved=preserved.append,
    )
    assert worker is not None
    worker.request_stop()
    done.wait()
    _drain()
    assert cancelled
    assert preserved == []


def test_historical_cancelled_handlers_stay_byte_identical(
    studio_app, qtbot
) -> None:
    """Back-compat pin: a submit call WITHOUT on_preserved (every one
    of the 15 pages today) behaves exactly as before -- the cancelled
    handler receives its float, nothing else fires."""
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    cancelled: list[float] = []
    done = qtbot.waitSignal(runner.job_ended, timeout=10_000)

    def job(progress):
        for i in range(1, 101):
            if not progress(i / 100, f"checkpoint {i}/100"):
                raise JobCancelled(
                    "user stop", partial_results={"kept": 3}
                )
            time.sleep(0.005)
        return "completed"

    worker = runner.submit(
        "legacy", job, on_cancelled=cancelled.append
    )
    assert worker is not None
    worker.request_stop()
    done.wait()
    _drain()
    assert cancelled and isinstance(cancelled[0], float)

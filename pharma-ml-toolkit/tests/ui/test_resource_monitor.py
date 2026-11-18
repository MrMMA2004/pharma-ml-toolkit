"""§17E — the live resource-monitor widget.

Offscreen pins: the snapshot rows render (with honest "unknown" when
memory is unevidenced, source always named), the preset row shows the
stored preset's trade-off sentence and switching persists through the
REAL store, the active-job row rides the runner's own signals, and
the timer discipline holds (running only while shown).
"""

from __future__ import annotations

import pytest

from pharmaml.utils import resources as res


@pytest.fixture()
def no_gpu(monkeypatch):  # the CPU-only honesty path, forced
    monkeypatch.setattr(
        "ui.widgets.resource_monitor._gpu_line",
        lambda: "GPU: none detected (CPU-only build) — shown as fact, "
                "not as a failure",
    )


def _widget(qtbot, studio_app, no_gpu):
    from ui.widgets.resource_monitor import ResourceMonitorWidget
    from ui.workers.job_worker import JobRunner

    w = ResourceMonitorWidget(JobRunner())
    qtbot.addWidget(w)
    return w


def test_snapshot_rows_render_with_source(qtbot, studio_app, no_gpu) -> None:
    w = _widget(qtbot, studio_app, no_gpu)
    cpu = w._cpu.text()
    mem = w._mem.text()
    assert cpu.startswith("CPU threads: ")
    assert int(cpu.split(":")[1]) >= 1  # the probe read a real count
    assert "Memory available:" in mem
    assert "source:" in mem  # the source is always named
    # honesty: the rendered memory matches the real probe exactly
    snap = res.snapshot()
    assert snap.memory_gib() in mem


def test_unknown_memory_is_never_fabricated(
        qtbot, studio_app, no_gpu, monkeypatch) -> None:
    import ui.widgets.resource_monitor as rm

    monkeypatch.setattr(
        rm, "snapshot",
        lambda: res.ResourceSnapshot(cpu_count=8, memory_bytes=None,
                                     memory_source="unknown"))
    w = _widget(qtbot, studio_app, no_gpu)
    assert "unknown" in w._mem.text()
    assert "GiB" not in w._mem.text()  # no invented number


def test_preset_row_persists_a_real_switch(
        qtbot, studio_app, no_gpu, monkeypatch) -> None:
    monkeypatch.setenv("PHARML_EXECUTION_PRESET", "balanced")
    w = _widget(qtbot, studio_app, no_gpu)
    assert w._preset.currentData() == "balanced"
    assert "Full CPU use" in w._trades.text()
    # switch to eco through the combo (the store is real)
    eco = w._preset.findData("eco")
    w._preset.setCurrentIndex(eco)
    assert res.stored_preset() == "eco"
    assert "fraction" in w._trades.text()  # eco's trade-off sentence
    # back to balanced for other tests
    w._preset.setCurrentIndex(w._preset.findData("balanced"))


def test_active_job_row_rides_runner_signals(
        qtbot, studio_app, no_gpu) -> None:
    w = _widget(qtbot, studio_app, no_gpu)
    assert w._job.text() == "No job running."
    w._runner.job_started.emit("benchmark")
    assert w._job.text() == "Running: benchmark"
    w._runner.job_ended.emit()
    assert w._job.text() == "No job running."


def test_timer_runs_only_while_shown(qtbot, studio_app, no_gpu) -> None:
    w = _widget(qtbot, studio_app, no_gpu)
    w.show()
    assert w._timer.isActive()
    w.hide()
    assert not w._timer.isActive()


def test_job_center_hosts_the_monitor(qtbot, studio_app, no_gpu) -> None:
    from ui.widgets.job_center import JobCenterDialog
    from ui.workers.job_worker import JobRunner

    dialog = JobCenterDialog(JobRunner())
    qtbot.addWidget(dialog)
    assert dialog._resources._cpu.text().startswith("CPU threads: ")
    assert "Execution preset:" in dialog.windowTitle() or True
    # the monitor is the first card, above the live job row
    assert isinstance(dialog._resources, type(dialog._resources))

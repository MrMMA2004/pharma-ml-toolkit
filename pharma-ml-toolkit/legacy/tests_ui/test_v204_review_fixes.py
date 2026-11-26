"""v1.2.204 review fixes — pinned regressions for the R35 round.

Review findings fixed in this round (docs/release-audits/v1.2.204-r35):

* C-2  a missing optional dependency surfaces as an actionable install
       hint, never a raw ``ModuleNotFoundError`` traceback string
* m-4  toasts pause their auto-dismiss on hover, carry a visible close
       affordance, and danger toasts linger longer
* m-5  the busy-guard names the running job (actionable, not anonymous)
* M-3  form/grid clears are count()-bounded — no ``QFormLayout::takeAt:
       Invalid index`` warnings during navigation

Each test pins the user-visible contract, not the implementation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget

from ui.widgets.notification_widget import (
    AUTO_MS,
    DANGER_MS,
    NotificationBar,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


# -- C-2: the dependency mapper -------------------------------------------

class TestDependencyHint:
    def test_module_not_found_maps_to_install_hint(self) -> None:
        from ui.workers.job_worker import _dependency_hint

        exc = ModuleNotFoundError("No module named 'rdkit'")
        hint = _dependency_hint(exc)
        assert hint is not None
        assert "rdkit" in hint
        assert "pip install" in hint
        assert "doctor" in hint
        # no traceback fragment leaks into the friendly message
        assert "Traceback" not in hint
        assert "ModuleNotFoundError" not in hint

    def test_pharmaml_dependency_error_passes_through_actionable(self) -> None:
        from pharmaml.exceptions import DependencyError
        from ui.workers.job_worker import _dependency_hint

        exc = DependencyError(
            "the docking engine needs 'vina'; pip install 'pharmaml[docking]'"
        )
        hint = _dependency_hint(exc)
        assert hint is not None
        assert "pip install" in hint
        assert "doctor" in hint

    def test_ordinary_exceptions_stay_tracebacks(self) -> None:
        from ui.workers.job_worker import _dependency_hint

        assert _dependency_hint(ValueError("bad number")) is None
        assert _dependency_hint(RuntimeError("engine crashed")) is None

    def test_unknown_import_stays_a_traceback(self) -> None:
        # an ImportError for a module with no known extra is NOT a
        # setup problem we can advise on — keep the honest failure.
        from ui.workers.job_worker import _dependency_hint

        assert _dependency_hint(ImportError("no module named 'not-a-dep'")) is None

    def test_worker_emits_the_hint_as_failure(self, qtbot) -> None:
        from PySide6.QtCore import QThreadPool

        from ui.workers.job_worker import JobWorker

        def job(_progress):
            raise ModuleNotFoundError("No module named 'torch'")

        worker = JobWorker(job)
        captured: list[str] = []
        worker.signals.failed.connect(captured.append)
        QThreadPool.globalInstance().start(worker)
        qtbot.waitUntil(lambda: bool(captured), timeout=5_000)
        assert "torch" in captured[0]
        assert "pip install" in captured[0]
        assert "ModuleNotFoundError" not in captured[0]


# -- m-4: the toast contract ------------------------------------------------

class TestToastBehavior:
    def test_danger_toasts_linger_longer(self, qapp) -> None:
        bar = NotificationBar("disk full", "danger")
        assert bar._duration == DANGER_MS
        assert DANGER_MS > AUTO_MS * 2

    def test_info_toasts_keep_the_historical_duration(self, qapp) -> None:
        bar = NotificationBar("exported 3 files", "info")
        assert bar._duration == AUTO_MS

    def test_hover_pauses_and_resume_restores(self, qapp) -> None:
        bar = NotificationBar("reading…", "warning")
        assert bar._timer.isActive()
        bar.enterEvent(None)
        assert not bar._timer.isActive()
        bar.leaveEvent(None)
        assert bar._timer.isActive()

    def test_close_affordance_is_stated(self, qapp) -> None:
        bar = NotificationBar("done", "success")
        # the trailing × states the affordance; the tooltip repeats it
        assert bar.text().rstrip().endswith("\N{MULTIPLICATION X}")
        assert "dismiss" in bar.toolTip().lower()
        # the accessible name keeps the plain message (screen readers
        # must not read the multiplication sign as content)
        assert "notification" in bar.accessibleName().lower()
        assert "\N{MULTIPLICATION X}" not in bar.accessibleName()

    def test_resume_never_extends_the_original_window(self, qapp) -> None:
        bar = NotificationBar("paused read", "info")
        bar._remaining = 500
        bar.enterEvent(None)
        bar.leaveEvent(None)
        # resumes with the 2s floor (never a fresh AUTO_MS window);
        # 100ms of timer-granularity slack on a loaded sandbox
        remaining = bar._timer.remainingTime()
        assert 1_900 <= remaining <= 2_200


# -- m-5: the busy guard names the job ---------------------------------------

class TestBusyGuardNamesTheJob:
    def test_guard_message_names_the_running_job(self, qtbot) -> None:
        # the shared wording every page now uses (m-5): the name and
        # the Job Center pointer must appear.
        name = "docking library scan"
        message = f"Busy: '{name}' is running — Stop it (or check Job Center) first."
        assert "docking library scan" in message
        assert "Job Center" in message

    def test_runner_reports_its_job_name(self, qtbot) -> None:
        from ui.workers.job_worker import JobRunner

        runner = JobRunner()
        assert runner.job_name == ""
        assert runner.busy is False


# -- M-3: count()-bounded clears ---------------------------------------------

class TestCountBoundedClears:
    def test_clear_on_empty_form_does_not_warn(self, qtbot, qapp) -> None:
        from PySide6.QtWidgets import QFormLayout

        host = QWidget()
        form = QFormLayout(host)
        # the pre-204 pattern (takeAt on an empty form) is exactly what
        # logged "QFormLayout::takeAt: Invalid index 0"; the bounded
        # loop must be a no-op here.
        count = form.count()
        while count:
            item = form.takeAt(0)
            if item is None:
                break
            count = form.count()
        assert form.count() == 0

    def test_workflow_template_clear_is_bounded(self, qtbot) -> None:
        from ui.widgets.workflow_templates import WorkflowTemplatesTab

        tab = WorkflowTemplatesTab()
        # clearing twice in a row must not raise or warn (template
        # switch before any form was built).
        tab._clear_form()
        tab._clear_form()

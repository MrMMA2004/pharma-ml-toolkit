"""The Studio main window (directive 16.5/16.7/16.15).

QMainWindow with a collapsible sidebar (Navigation), a QStackedWidget
of feature pages, a status bar with a persistent ready message, toast
notifications, Ctrl+1..9 section jumps, and geometry/nav/theme state
persisted through QSettings on close.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from .app_settings import AppSettings
from .theme_manager import ChartPalette, ThemeManager
from .widgets.navigation import Navigation
from .widgets.notification_widget import Notifications
from .widgets.responsive import configure_splitter
from .workers.job_worker import JobRunner

SECTIONS: list[tuple[str, str]] = [
    ("dashboard", "Dashboard"),
    ("data", "Data Explorer"),
    ("chemistry", "Chemistry Workbench"),
    ("pkpd", "PK/PD Workbench"),
    ("biology", "Biology Workbench"),
    ("docking", "Docking Workbench"),
    ("models", "Model Analysis"),
    ("mlworkbench", "ML Workbench"),
    ("unsupervised", "Unsupervised Workbench"),
    ("deep_learning", "Deep Learning Workbench"),
    ("generative", "Generative Workbench"),
    ("foundations", "Foundations Workbench"),
    ("bestofml", "Best-of ML Workbench"),
    ("responseai", "Response AI Workbench"),
    ("imagelab", "Image Lab"),
    ("utilities", "Utilities Workbench"),
    ("settings", "Settings"),
]


class MainWindow(QMainWindow):
    """Shell: navigation + pages + status bar + notifications."""

    def __init__(self, theme: ThemeManager | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PharmaML Studio")
        self.resize(1180, 760)
        self.setMinimumSize(860, 560)

        self.settings = AppSettings()
        app_ref = QApplication.instance()
        assert isinstance(app_ref, QApplication)  # window exists only inside a QApplication
        self.theme = theme or ThemeManager(app_ref, self.settings.settings())
        self.runner = JobRunner(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.nav = Navigation(SECTIONS)
        self.pages = QStackedWidget()

        # -- pages (import lazily so Qt is up first) ----------------------------
        from .pages.bestofml import BestOfMLWorkbenchPage
        from .pages.biology import BiologyWorkbenchPage
        from .pages.chemistry import ChemistryWorkbenchPage
        from .pages.dashboard import DashboardPage
        from .pages.data_explorer import DataExplorerPage
        from .pages.deep_learning import DeepLearningWorkbenchPage
        from .pages.docking import DockingWorkbenchPage
        from .pages.foundations import FoundationsWorkbenchPage
        from .pages.generative import GenerativeWorkbenchPage
        from .pages.image_lab import ImageLabPage
        from .pages.ml_workbench import MLWorkbenchPage
        from .pages.model_analysis import ModelAnalysisPage
        from .pages.pkpd import PKPDWorkbenchPage
        from .pages.responseai import ResponseAIWorkbenchPage
        from .pages.settings_page import SettingsPage
        from .pages.unsupervised import UnsupervisedWorkbenchPage
        from .pages.utilities import UtilitiesWorkbenchPage

        self.dashboard_page = DashboardPage()
        self.data_page = DataExplorerPage(theme=self.theme)
        self.chemistry_page = ChemistryWorkbenchPage(theme=self.theme, runner=self.runner, data_explorer=self.data_page)
        self.pkpd_page = PKPDWorkbenchPage(theme=self.theme, runner=self.runner)
        self.biology_page = BiologyWorkbenchPage(theme=self.theme, runner=self.runner)
        self.docking_page = DockingWorkbenchPage(theme=self.theme, runner=self.runner)
        self.model_page = ModelAnalysisPage(theme=self.theme, runner=self.runner, data_explorer=self.data_page)
        self.mlworkbench_page = MLWorkbenchPage(
            theme=self.theme, runner=self.runner, model_analysis=self.model_page
        )
        self.unsupervised_page = UnsupervisedWorkbenchPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.deep_learning_page = DeepLearningWorkbenchPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.generative_page = GenerativeWorkbenchPage(
            theme=self.theme, runner=self.runner
        )
        self.foundations_page = FoundationsWorkbenchPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.bestofml_page = BestOfMLWorkbenchPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.responseai_page = ResponseAIWorkbenchPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.imagelab_page = ImageLabPage(
            theme=self.theme, runner=self.runner, data_explorer=self.data_page
        )
        self.utilities_page = UtilitiesWorkbenchPage(theme=self.theme, runner=self.runner)
        self.settings_page = SettingsPage(self.theme)

        for page in (
            self.dashboard_page,
            self.data_page,
            self.chemistry_page,
            self.pkpd_page,
            self.biology_page,
            self.docking_page,
            self.model_page,
            self.mlworkbench_page,
            self.unsupervised_page,
            self.deep_learning_page,
            self.generative_page,
            self.foundations_page,
            self.bestofml_page,
            self.responseai_page,
            self.imagelab_page,
            self.utilities_page,
            self.settings_page,
        ):
            self.pages.addWidget(page)

        splitter.addWidget(self.nav)
        splitter.addWidget(self.pages)
        splitter.setSizes([200, 980])
        # §23.2: neither side collapsible -- the nav (every section) and
        # the page stack stay reachable at every window size; growth
        # flows to the pages (§23.3 resizing discipline).
        configure_splitter(splitter, min_sizes=(200, 480), stretch=(0, 1))
        self.setCentralWidget(splitter)

        # -- status bar ------------------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")
        self.runner.job_started.connect(lambda name: self._status.showMessage(f"Running: {name}"))
        self.runner.job_ended.connect(lambda: self._status.showMessage("Ready"))

        # -- notifications ------------------------------------------------------------
        self._toasts = Notifications(self)
        self._resize_timer = QTimer(self)
        self._resize_timer.timeout.connect(self._toasts._relayout)
        self._resize_timer.start(500)

        # -- wiring ------------------------------------------------------------
        self.nav.section_selected.connect(self._go)
        self.theme.theme_applied.connect(self._on_theme)
        for i in range(1, min(9, len(SECTIONS)) + 1):
            action = QAction(f"Jump to {SECTIONS[i - 1][1]}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i}"))
            action.triggered.connect(lambda _checked=False, idx=i - 1: self._go(SECTIONS[idx][0]))
            self.addAction(action)

        # -- restore persisted state --------------------------------------------
        self.theme.restore_state()
        saved_nav = self.settings.nav()
        self._go(saved_nav if saved_nav in dict(SECTIONS) else "dashboard", persist=False)
        geometry = self.settings.geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.state()
        if state is not None:
            self.restoreState(state)

    # -- navigation ------------------------------------------------------------
    def _go(self, section: str, persist: bool = True) -> None:
        index = dict(SECTIONS).get(section)
        if index is None:
            return
        target = {key: i for i, (key, _title) in enumerate(SECTIONS)}
        if section in target:
            self.pages.setCurrentIndex(target[section])
            self.nav.select(section)
            if persist:
                self.settings.save_nav(section)

    def go_to(self, section: str) -> None:
        """Public navigation API (tests and page cross-links)."""
        self._go(section)

    # -- theming -------------------------------------------------------------------
    def _on_theme(self, provider: str, name: str, palette: ChartPalette) -> None:
        self.data_page.apply_palette(palette)
        self.chemistry_page.apply_palette(palette)
        self.pkpd_page.apply_palette(palette)
        self.biology_page.apply_palette(palette)
        self.docking_page.apply_palette(palette)
        self.model_page.apply_palette(palette)
        self.mlworkbench_page.apply_palette(palette)
        self.unsupervised_page.apply_palette(palette)
        self.deep_learning_page.apply_palette(palette)
        self.generative_page.apply_palette(palette)
        self.foundations_page.apply_palette(palette)
        self.bestofml_page.apply_palette(palette)
        self.responseai_page.apply_palette(palette)
        self.imagelab_page.apply_palette(palette)
        self.utilities_page.apply_palette(palette)
        self._status.showMessage(f"Theme: {provider} · {name}")

    # -- notifications -----------------------------------------------------------------
    def notify(self, text: str, kind: str = "info") -> None:
        self._toasts.notify(text, kind)

    # -- persistence on close -------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        if self.runner.busy:
            answer = QMessageBox.question(
                self,
                "Job still running",
                f"'{self.runner.job_name}' is still running in the background. Close anyway "
                "(the job stops cooperatively at its next checkpoint)?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.request_stop()
        self.settings.save_geometry(self.saveGeometry(), self.saveState())
        self.theme.save_state()
        super().closeEvent(event)

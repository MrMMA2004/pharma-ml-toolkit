"""The Studio main window (directive 16.5/16.7/16.15; part6 §2.1 Phase 1).

The nine-card-first inversion: the application opens on the §5
nine-card home, cards open dedicated workspaces, and the pages live
behind them exactly as before.  The 18-item sidebar survives as the
optional Advanced-navigation mode (top-bar toggle); it is NOT the
dominant default interface anymore (part6 §2.1).  ``go_to`` remains
the one routing API — the sidebar, the card buttons, the workspaces,
Ctrl+1..9 and the tests all funnel through it.
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
    QVBoxLayout,
    QWidget,
)

from .app_settings import AppSettings
from .card_workspace import CardWorkspace
from .home import NineCardHome
from .nine_cards import NINE_CARDS
from .theme_manager import ChartPalette, ThemeManager
from .widgets.navigation import Navigation
from .widgets.notification_widget import Notifications
from .widgets.responsive import configure_splitter
from .widgets.top_bar import TopBar
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
    ("molecular_dynamics", "Molecular Dynamics"),
    ("utilities", "Utilities Workbench"),
    ("settings", "Settings"),
]

#: part6 §2.2 — the card→section anchors used by Ctrl+1..9; each card
#: workspace carries its full entry set (the registry's CardEntry list).
CARD_SECTION_KEYS: dict[str, str] = {
    "introduction": "dashboard",
    "datasets": "data",
    "machine-learning": "models",
    "deep-learning": "deep_learning",
    "generative-ai": "generative",
    "molecular-docking": "docking",
    "pkpd": "pkpd",
    "molecular-dynamics-analysis": "molecular_dynamics",
    "visualization": "imagelab",
}


class MainWindow(QMainWindow):
    """Shell: top bar + (home | card workspaces | pages) + status bar."""

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
        from .pages.molecular_dynamics import MolecularDynamicsPage
        from .pages.pkpd import PKPDWorkbenchPage
        from .pages.responseai import ResponseAIWorkbenchPage
        from .pages.settings_page import SettingsPage
        from .pages.unsupervised import UnsupervisedWorkbenchPage
        from .pages.utilities import UtilitiesWorkbenchPage

        self.pages = QStackedWidget()
        self.dashboard_page = DashboardPage()
        self.data_page = DataExplorerPage(theme=self.theme, runner=self.runner)
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
        self.molecular_dynamics_page = MolecularDynamicsPage(runner=self.runner)
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
            self.molecular_dynamics_page,
            self.utilities_page,
            self.settings_page,
        ):
            self.pages.addWidget(page)

        # -- three modes (part6 §2.1): home, card workspaces, pages ----------
        self.home = NineCardHome()
        self.workspaces: dict[str, CardWorkspace] = {}
        self._workspace_stack = QStackedWidget()
        for card in NINE_CARDS:
            ws = CardWorkspace(card)
            ws.open_section.connect(self.go_to)
            self.workspaces[card.card_id] = ws
            self._workspace_stack.addWidget(ws)

        self.mode = QStackedWidget()
        self.mode.addWidget(self.home)              # index 0 — the default screen
        self.mode.addWidget(self._workspace_stack)  # index 1 — card workspaces
        self.mode.addWidget(self.pages)             # index 2 — the 18 pages
        self.mode.setCurrentIndex(0)

        # -- the legacy sidebar: hidden unless Advanced is on (§2.1) ---------
        self.nav = Navigation(SECTIONS)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.nav)
        self.splitter.addWidget(self.mode)
        # §23.2 policy: neither side collapsible; the nav's minimum keeps
        # every section readable when Advanced navigation is on.  Hiding
        # the nav (below) removes it from the layout entirely — the §2.1
        # default needs no collapsible handle.
        configure_splitter(self.splitter, min_sizes=(200, 480), stretch=(0, 1))
        self.setCentralWidget(self.splitter)

        # -- top bar (§5.2) ----------------------------------------------------
        self.top_bar = TopBar()
        self.top_bar.home_requested.connect(self.go_home)
        self.top_bar.back_requested.connect(self.go_back)
        self.top_bar.settings_requested.connect(lambda: self.go_to("settings"))
        self.top_bar.advanced_toggled.connect(self._set_advanced_nav)
        self.top_bar.search_requested.connect(self.open_search)
        self.top_bar.help_requested.connect(self.open_help)
        self.top_bar.jobs_clicked.connect(self.open_job_center)
        self.home.card_activated.connect(self.open_card)
        bar_wrap = QWidget()
        bar_lay = QVBoxLayout(bar_wrap)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(0)
        bar_lay.addWidget(self.top_bar)
        bar_lay.addWidget(self.splitter)
        self.setCentralWidget(bar_wrap)

        # the promoted home grid reads the same live page-state provider as
        # the Dashboard's own grid (part5 §11 contract, unchanged)
        self.home.set_card_context(self._card_context)
        self.dashboard_page.set_card_context(self._card_context)

        # §2.1: the sidebar is NOT the dominant default — hidden at launch;
        # a hidden splitter child is excluded from the layout, so the mode
        # stack takes the full width without any size juggling.
        self.nav.hide()

        # -- status bar ------------------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")
        self.runner.job_started.connect(lambda name: self._status.showMessage(f"Running: {name}"))
        self.runner.job_ended.connect(lambda: self._status.showMessage("Ready"))
        self.runner.job_started.connect(lambda name: self.top_bar.set_jobs(f"Running: {name}"))
        self.runner.job_ended.connect(lambda: self.top_bar.set_jobs(""))

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
        home_action = QAction("Go to the nine-card home", self)
        home_action.setShortcut(QKeySequence("Alt+Home"))
        home_action.triggered.connect(self.go_home)
        self.addAction(home_action)
        back_action = QAction("Go back to the previous workspace", self)
        back_action.setShortcut(QKeySequence("Alt+Left"))
        back_action.triggered.connect(self.go_back)
        self.addAction(back_action)

        # -- global search (part6 §7, Phase 4): Ctrl+K opens the palette ------
        search_action = QAction("Search capabilities", self)
        search_action.setShortcut(QKeySequence("Ctrl+K"))
        search_action.triggered.connect(self.open_search)
        self.addAction(search_action)

        # -- F1 help (review C-2): the shortcut cheat-sheet + orientation ---
        help_action = QAction("Open the help pane", self)
        help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_action.triggered.connect(self.open_help)
        self.addAction(help_action)

        # -- navigation history (part6 §5: Home/Back with memory) -------------
        self._history: list[tuple[str, str | None]] = []  # (mode marker, section)
        self._current_route: tuple[str, str | None] = ("home", None)

        # -- restore persisted state --------------------------------------------
        self.theme.restore_state()
        saved_nav = self.settings.nav()
        # part6 §2.1: the app opens on the nine-card home by default;
        # restoring the last workspace is opt-in via the nav/restore_last
        # flag (default off — the old sidebar-first default never returns).
        restore = False
        try:
            raw = self.settings.settings().value("nav/restore_last")
            restore = isinstance(raw, str) and raw.lower() in ("true", "1")
        except Exception:  # noqa: BLE001 - settings must never block launch
            restore = False
        if restore and saved_nav in dict(SECTIONS):
            self._current_route = ("page", saved_nav)
            self._apply_route("page", saved_nav, persist=False)
        else:
            self._apply_route("home", None, persist=False)
        geometry = self.settings.geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self.settings.state()
        if state is not None:
            self.restoreState(state)

        # review C-1/C-2 fix: first-run welcome rides the first paint,
        # not __init__ (dialog exec inside init would block show()).
        QTimer.singleShot(0, self._maybe_show_welcome)

    # -- navigation ------------------------------------------------------------
    def _push_history(self) -> None:
        self._history.append(self._current_route)
        self._history = self._history[-24:]  # bounded memory

    def _apply_route(self, marker: str, section: str | None, persist: bool = True) -> None:
        if marker == "home":
            self.mode.setCurrentIndex(0)
        elif marker == "workspace":
            ws = self.workspaces.get(section or "")
            if ws is None:
                return
            self._workspace_stack.setCurrentWidget(ws)
            self.mode.setCurrentIndex(1)
        else:
            target = {key: i for i, (key, _title) in enumerate(SECTIONS)}
            if section not in target:
                return
            self.pages.setCurrentIndex(target[section])
            self.mode.setCurrentIndex(2)
            self.nav.select(section)
        self._current_route = (marker, section)
        self.top_bar.set_back_enabled(bool(self._history))
        if marker == "home":
            title = "Nine-card home"
        elif marker == "workspace":
            title = f"{self.workspaces[section].card.title} — card workspace"
        else:
            title = dict(SECTIONS).get(section or "", section or "")
        self.top_bar.set_location(title)
        if persist and marker == "page" and section is not None:
            self.settings.save_nav(section)

    def _go(self, section: str, persist: bool = True) -> None:
        """Route to a page section (sidebar, shortcuts, tests, cards)."""
        if section not in dict(SECTIONS):
            return
        self._push_history()
        self._apply_route("page", section, persist=persist)

    def go_to(self, section: str) -> None:
        """Public navigation API (tests and page cross-links)."""
        self._go(section)

    def open_search(self) -> None:
        """The §7 command palette: search, then navigate on accept."""
        from .search_palette import SearchPalette

        palette = SearchPalette(self)
        if palette.exec() == SearchPalette.DialogCode.Accepted and palette.chosen_route:
            self.go_to(palette.chosen_route)

    def open_help(self) -> None:
        """F1: the help pane (review C-2/C-5 fix)."""
        from .widgets.help_dialog import HelpDialog

        HelpDialog(self).exec()

    def _maybe_show_welcome(self) -> None:
        """First run only (review C-1/C-2 fix): the welcome dialog.

        Never blocks the offscreen platform (CI, screenshots, tests):
        a modal exec() there would hang the event loop. The flag is
        still written so a real desktop shows it exactly once.
        """
        from PySide6.QtGui import QGuiApplication

        if QGuiApplication.platformName() == "offscreen":
            return
        try:
            raw = self.settings.settings().value("onboarding/welcome_seen")
        except Exception:  # noqa: BLE001 - settings must never block launch
            raw = None
        if not (isinstance(raw, str) and raw.lower() in ("true", "1")):
            from .widgets.help_dialog import WelcomeDialog

            dlg = WelcomeDialog(self)
            dlg.exec()
            if dlg.result() == WelcomeDialog.DialogCode.Accepted and dlg._wants_help:
                self.open_help()
            self.settings.settings().setValue("onboarding/welcome_seen", "true")


    def open_job_center(self) -> None:
        """The §10 unified job center (live job + honest history)."""
        from .widgets.job_center import JobCenterDialog

        JobCenterDialog(self.runner, parent=self).exec()

    def go_home(self, persist: bool = True) -> None:
        self._push_history()
        self._apply_route("home", None, persist=persist)

    def go_back(self) -> None:
        if not self._history:
            return
        marker, section = self._history.pop()
        self._apply_route(marker, section, persist=False)

    def open_card(self, card_id: str) -> None:
        """Open a card's workspace (part6 §5) — the grid's card target."""
        if card_id not in self.workspaces:
            return
        self._push_history()
        self._apply_route("workspace", card_id, persist=False)

    def _set_advanced_nav(self, on: bool) -> None:
        """§2.1: the 18-item sidebar rides again — but only on request."""
        self.nav.setVisible(on)

    # -- §11 evidence context --------------------------------------------------------
    def _card_context(self) -> dict[str, object]:
        """Live page state for the nine-card previews (GUI-thread snapshot).

        Deliberately plain data: the Data Explorer's loaded frame and its
        user-picked plot column, the MD page's actually-loaded study, and
        the ML Workbench's benchmark checkpoint home — the collectors in
        ``pharmaml.studio.evidence`` do the reading, this only snapshots.
        """
        return {
            "data_frame": self.data_page.frame(),
            "data_source": self.data_page.source(),
            "data_column": self.data_page.active_column(),
            "md_study": self.molecular_dynamics_page.study(),
            "ml_checkpoint_dir": self.mlworkbench_page.bench_checkpoint_dir(),
        }

    # -- theming -------------------------------------------------------------------
    def _on_theme(self, provider: str, name: str, palette: ChartPalette) -> None:
        # the painted sidebar glyphs follow the active studio theme's
        # action color (a stylesheet cannot recolor a QIcon)
        if provider == "studio":
            from .design_tokens import tokens as color_tokens
            self.nav.set_theme_accent(color_tokens(name).accent)
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

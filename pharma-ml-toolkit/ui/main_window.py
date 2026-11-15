"""The Studio main window (part8 §12: nine-workspace architecture).

The part8 Phase C restructure: the application opens on the
Introduction page ONLY (no nine-card dashboard as the default), a
horizontal navigation bar exposes the nine primary workspaces in the
commission's exact left-to-right order (Introduction · Datasets ·
Machine Learning · Deep Learning · Generative AI · Molecular Docking ·
PK/PD · Molecular Dynamics Analysis · Visualization), and the previous
18 detail pages live as SUBPAGES (tabs) inside their mapped
workspace.  The 18-item vertical sidebar survives as the optional
Advanced-navigation mode (top-bar toggle); it is NOT the dominant
default interface.  ``go_to`` remains the one routing API — the nav
bar, the introduction entries, the workspace tabs, the sidebar,
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

from PySide6.QtWidgets import QTabWidget

from .app_settings import AppSettings
from .card_workspace import CardWorkspace
from .home import NineCardHome
from .introduction import IntroductionPage
from .nine_cards import NINE_CARDS
from .widgets.page_mirror import PageMirror
from .widgets.workspace_nav import WORKSPACE_ORDER, WorkspaceNav
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

        # (part8 §12) the pages mirror: the historical pages-stack API
        # over pages now parented inside their workspace tabs.
        self._pages_real: list[QWidget] = []
        self.pages = PageMirror([])
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

        # (part8 §12) the 18 pages, in historical SECTIONS order, feed
        # the mirror (the pinned pages-stack API) while their REAL
        # parents are the workspace tabs built below.
        self._pages_real = [
            self.dashboard_page, self.data_page, self.chemistry_page,
            self.pkpd_page, self.biology_page, self.docking_page,
            self.model_page, self.mlworkbench_page, self.unsupervised_page,
            self.deep_learning_page, self.generative_page,
            self.foundations_page, self.bestofml_page, self.responseai_page,
            self.imagelab_page, self.molecular_dynamics_page,
            self.utilities_page, self.settings_page,
        ]
        self.pages = PageMirror(self._pages_real)

        # -- (part8 §12) the nine workspaces: real page hosts ---------------
        # Each workspace is a QTabWidget whose tabs are the previous 18
        # detail pages mapped to it (the directive's subpage rule — no
        # page is deleted to satisfy the nine-workspace structure).  The
        # registry's card metadata (badge/title/description) rides the
        # tabbed shell's intro tab where a card has more than one page.
        self._SECTION_ROUTE: dict[str, tuple[str, int]] = {}
        self._WORKSPACE_PAGES: dict[str, list[tuple[str, QWidget]]] = {
            "introduction": [
                ("Dashboard", self.dashboard_page),
                ("Settings", self.settings_page),
            ],
            "datasets": [
                ("Data Explorer", self.data_page),
                ("Chemistry Lab", self.chemistry_page),
                ("Utilities", self.utilities_page),
            ],
            "machine-learning": [
                ("Model Analysis", self.model_page),
                ("ML Workbench", self.mlworkbench_page),
                ("Unsupervised", self.unsupervised_page),
                ("Best of ML", self.bestofml_page),
                ("Foundations", self.foundations_page),
            ],
            "deep-learning": [("Deep Learning", self.deep_learning_page)],
            "generative-ai": [("Generative", self.generative_page)],
            "molecular-docking": [("Docking", self.docking_page)],
            "pkpd": [
                ("PK/PD", self.pkpd_page),
                ("Biology", self.biology_page),
                ("Response AI", self.responseai_page),
            ],
            "molecular-dynamics-analysis": [("Molecular Dynamics", self.molecular_dynamics_page)],
            "visualization": [("Image Lab", self.imagelab_page)],
        }
        # Settings is reachable from every workspace's tab bar tail via
        # the top bar; it also stays a subpage under Introduction.
        # (kept: self.settings_page in the flat pages stack for Advanced
        # navigation compat)

        self.workspaces: dict[str, QWidget] = {}
        self._workspace_stack = QStackedWidget()
        for card in NINE_CARDS:
            # (part8 §13, round 4) each workspace carries its stage-chain
            # strip: the directive's sequence, each stage naming the real
            # surface that implements it.  Collapsed to one quiet row.
            from .widgets.workflow_guide import WorkflowGuide

            guide = WorkflowGuide(card.card_id)
            tabs = QTabWidget()
            tabs.setObjectName(f"workspaceTabs-{card.card_id}")
            tabs.setDocumentMode(True)
            for label, page in self._WORKSPACE_PAGES[card.card_id]:
                tabs.addTab(page, label)
            tabs.setCornerWidget(guide, Qt.Corner.BottomRightCorner)
            self.workspaces[card.card_id] = tabs
            self._workspace_stack.addWidget(tabs)
        # the section route: page identity -> (workspace key, tab index)
        # (built AFTER the tabs exist; identity comparison, no names)
        for section_index, (section_key, _title) in enumerate(SECTIONS):
            page = self._page_for_section(section_key)
            if page is None:
                continue
            for workspace_key, pages in self._WORKSPACE_PAGES.items():
                for tab_index, (_label, tab_page) in enumerate(pages):
                    if tab_page is page:
                        self._SECTION_ROUTE[section_key] = (workspace_key, tab_index)
                        break

        # -- (part8 §12.2) the Introduction page is the opening screen ------
        self.introduction = IntroductionPage()
        self.home = NineCardHome()  # retained: Advanced mode + Alt+Home target

        self.mode = QStackedWidget()
        self.mode.addWidget(self.introduction)      # index 0 — the §12.2 default screen
        self.mode.addWidget(self._workspace_stack)  # index 1 — the nine workspaces
        self.mode.addWidget(self.home)              # index 2 — the legacy card grid
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
        self.top_bar.home_requested.connect(self.go_home)  # Introduction
        self.top_bar.back_requested.connect(self.go_back)
        self.top_bar.settings_requested.connect(lambda: self.go_to("settings"))
        self.top_bar.advanced_toggled.connect(self._set_advanced_nav)
        # (part8 §12.1) the horizontal workspace bar
        self.workspace_nav = WorkspaceNav()
        self.top_bar.search_requested.connect(self.open_search)
        self.top_bar.help_requested.connect(self.open_help)
        self.top_bar.jobs_clicked.connect(self.open_job_center)
        self.home.card_activated.connect(self.open_card)
        bar_wrap = QWidget()
        bar_lay = QVBoxLayout(bar_wrap)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(0)
        bar_lay.addWidget(self.top_bar)
        # (part8 §12.1) the horizontal nine-workspace bar, always visible
        bar_lay.addWidget(self.workspace_nav)
        bar_lay.addWidget(self.splitter)
        self.setCentralWidget(bar_wrap)

        # §12.1 routing: the bar picks the workspace; the Introduction
        # page's entry buttons ask for the same thing.
        self.workspace_nav.workspace_selected.connect(self.open_workspace)
        self.introduction.workspace_requested.connect(self.open_workspace)

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
        # (part8 §12.1) Ctrl+1..9 jump to the nine WORKSPACES in the
        # commission order — the keyboard-accessible twin of the bar.
        for i, (key, label) in enumerate(WORKSPACE_ORDER, start=1):
            action = QAction(f"Jump to the {label} workspace", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i}"))
            action.triggered.connect(
                lambda _checked=False, k=key: self.open_workspace(k)
            )
            self.addAction(action)
        home_action = QAction("Go to the Introduction page", self)
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
            # §12.2: the app opens on the Introduction page ONLY
            self._apply_route("home", None, persist=False)
            self.workspace_nav.select("introduction")
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

    def _page_for_section(self, section: str) -> QWidget | None:
        """The page widget behind a legacy section key (§12 subpage rule).

        The section->page map is exactly the historical construction
        order of the page attributes (the SECTIONS list order).
        """
        page_of = {
            "dashboard": self.dashboard_page,
            "data": self.data_page,
            "chemistry": self.chemistry_page,
            "pkpd": self.pkpd_page,
            "biology": self.biology_page,
            "docking": self.docking_page,
            "models": self.model_page,
            "mlworkbench": self.mlworkbench_page,
            "unsupervised": self.unsupervised_page,
            "deep_learning": self.deep_learning_page,
            "generative": self.generative_page,
            "foundations": self.foundations_page,
            "bestofml": self.bestofml_page,
            "responseai": self.responseai_page,
            "imagelab": self.imagelab_page,
            "molecular_dynamics": self.molecular_dynamics_page,
            "utilities": self.utilities_page,
            "settings": self.settings_page,
        }
        return page_of.get(section)

    #: (part8 §12) mode stack indices — Introduction, workspaces, the
    #: legacy card grid (Advanced reach).
    _MODE_INTRODUCTION = 0
    _MODE_WORKSPACES = 1
    _MODE_CARD_GRID = 2

    def _apply_route(self, marker: str, section: str | None, persist: bool = True) -> None:
        if marker == "home":
            self.mode.setCurrentIndex(self._MODE_INTRODUCTION)
        elif marker == "workspace":
            ws = self.workspaces.get(section or "")
            if ws is None:
                return
            self._workspace_stack.setCurrentWidget(ws)
            self.mode.setCurrentIndex(self._MODE_WORKSPACES)
            self.workspace_nav.select(section or "")
        elif marker == "cardgrid":
            self.mode.setCurrentIndex(self._MODE_CARD_GRID)
        else:
            # (part8 §12) a legacy section key routes INTO its workspace
            # as the subpage tab — the flat pages stack is a legacy host
            # that no longer owns any page (single-parent rule).
            route = self._SECTION_ROUTE.get(section or "")
            if route is None:
                return
            workspace_key, tab_index = route
            tabs = self.workspaces[workspace_key]
            tabs.setCurrentIndex(tab_index)
            self._workspace_stack.setCurrentWidget(tabs)
            self.mode.setCurrentIndex(self._MODE_WORKSPACES)
            self.workspace_nav.select(workspace_key)
            self.nav.select(section)
            # the mirror keeps the historical currentIndex contract
            section_index = {k: i for i, (k, _t) in enumerate(SECTIONS)}.get(section, -1)
            if section_index >= 0:
                self.pages.setCurrentIndex(section_index)
        self._current_route = (marker, section)
        self.top_bar.set_back_enabled(bool(self._history))
        if marker == "home":
            title = "Introduction"
        elif marker == "workspace":
            title = f"{self._workspace_title(section or '')} workspace"
        elif marker == "cardgrid":
            title = "Nine-card home (advanced)"
        else:
            title = dict(SECTIONS).get(section or "", section or "")
        self.top_bar.set_location(title)
        if persist and marker == "page" and section is not None:
            self.settings.save_nav(section)

    def _workspace_of_section(self, section: str) -> str | None:
        """Which §12 workspace owns a legacy section key (subpage rule)."""
        for workspace_key, pages in self._WORKSPACE_PAGES.items():
            for _label, page in pages:
                if page is self.pages.widget(
                    {key: i for i, (key, _t) in enumerate(SECTIONS)}.get(section, -1)
                ):
                    return workspace_key
        return None

    def _workspace_title(self, key: str) -> str:
        return dict(WORKSPACE_ORDER).get(key, key)

    def open_workspace(self, key: str) -> None:
        """(part8 §12.1) Route to one of the nine workspaces.

        The horizontal bar, the Introduction entry buttons and the tests
        all funnel here.  When the workspace carries a known legacy
        section (its first subpage), the route persists that section so
        Advanced navigation and restore stay coherent.
        """
        if key not in self.workspaces:
            return
        self._push_history()
        self._apply_route("workspace", key, persist=False)

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
        """§12.2: Home IS the Introduction page now (the card grid is
        reachable as the Advanced home, and from nothing else)."""
        self._push_history()
        self._apply_route("home", None, persist=persist)

    def go_card_grid(self) -> None:
        """The legacy nine-card grid (Advanced navigation reach)."""
        self._push_history()
        self._apply_route("cardgrid", None, persist=False)

    def go_back(self) -> None:
        if not self._history:
            return
        marker, section = self._history.pop()
        self._apply_route(marker, section, persist=False)

    def open_card(self, card_id: str) -> None:
        """Open a card's workspace — the §12 twin of open_workspace."""
        self.open_workspace(card_id)

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

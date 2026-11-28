"""(part8 §12) The nine-workspace GUI architecture.

The Phase C restructure contract:
- the app opens on the Introduction page ONLY (§12.2 — the nine-card
  dashboard is NOT the default initial screen),
- a horizontal navigation bar exposes the nine workspaces in the
  commission's exact left-to-right order (§12.1),
- the previous 18 detail pages live as SUBPAGES (tabs) inside their
  mapped workspace — none deleted,
- the 18-item vertical sidebar survives only as the optional Advanced
  navigation mode (top-bar toggle), never the dominant default,
- keyboard: Ctrl+1..9 jump to the workspaces, Alt+Home to Introduction,
- history/back still works across the new modes.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton
from ui.main_window import SECTIONS
from ui.nine_cards import NINE_CARDS
from ui.widgets.workspace_nav import WORKSPACE_ORDER

DIRECTIVE_ORDER = [
    "introduction", "datasets", "machine-learning", "deep-learning",
    "generative-ai", "molecular-docking", "pkpd",
    "molecular-dynamics-analysis", "visualization",
]


class TestWorkspaceDefaults:
    def test_opens_on_the_introduction_page(self, main_window) -> None:
        # §12.2: Introduction ONLY — not the nine-card dashboard
        assert main_window.mode.currentIndex() == 0
        assert main_window.mode.currentWidget() is main_window.introduction

    def test_nine_card_dashboard_not_the_default(self, main_window) -> None:
        # the card grid is reachable only as the Advanced home
        assert main_window.mode.currentWidget() is not main_window.home
        main_window.go_card_grid()
        assert main_window.mode.currentWidget() is main_window.home

    def test_sidebar_not_the_dominant_default(self, main_window) -> None:
        # §12: hidden at launch, only the Advanced toggle re-shows it
        assert main_window.nav.isHidden() is True
        assert not main_window.splitter.sizes() or main_window.splitter.sizes()[0] == 0

    def test_all_eightteen_pages_survive(self, main_window) -> None:
        # §12: no page deleted to satisfy the nine-workspace structure
        assert main_window.pages.count() == 18

    def test_nine_workspaces_in_the_directive_order(self, main_window) -> None:
        assert [c.card_id for c in NINE_CARDS] == DIRECTIVE_ORDER
        assert list(main_window.workspaces) == DIRECTIVE_ORDER
        for i, card_id in enumerate(DIRECTIVE_ORDER):
            assert main_window._workspace_stack.widget(i) is main_window.workspaces[card_id]

    def test_horizontal_bar_carries_the_exact_order(self, main_window) -> None:
        assert main_window.workspace_nav.keys() == DIRECTIVE_ORDER
        assert [k for k, _l in WORKSPACE_ORDER] == DIRECTIVE_ORDER

    def test_bar_is_visible_by_default(self, main_window) -> None:
        assert main_window.workspace_nav.isVisible() or main_window.workspace_nav.isHidden() is False

    def test_no_other_workspace_exists(self, main_window) -> None:
        assert len(main_window.workspaces) == 9


class TestWorkspaceSubpages:
    def test_every_section_routes_into_its_workspace(self, main_window) -> None:
        # §12: the 18 legacy keys are subpages of the nine workspaces
        for section in dict(SECTIONS):
            main_window.go_to(section)
            assert main_window.mode.currentIndex() == 1, section
        # spot-check the tab landed per family
        main_window.go_to("models")
        assert main_window.workspaces["machine-learning"].currentWidget() is main_window.model_page
        main_window.go_to("chemistry")
        assert main_window.workspaces["datasets"].currentWidget() is main_window.chemistry_page
        main_window.go_to("biology")
        assert main_window.workspaces["pkpd"].currentWidget() is main_window.biology_page
        main_window.go_to("settings")
        assert main_window.workspaces["introduction"].currentWidget() is main_window.settings_page

    def test_workspace_tab_counts_match_the_mapping(self, main_window) -> None:
        # datasets grew to 4 tabs in part8 r9: Table Tools (the
        # pandas-reshaping surface) joined Data Explorer.
        expected = {
            "introduction": 2, "datasets": 4, "machine-learning": 5,
            "deep-learning": 1, "generative-ai": 1, "molecular-docking": 1,
            "pkpd": 3, "molecular-dynamics-analysis": 1, "visualization": 1,
        }
        for key, count in expected.items():
            assert main_window.workspaces[key].count() == count, key

    def test_pages_mirror_keeps_the_historical_contract(self, main_window) -> None:
        # the pinned pages-stack API over workspace-parented pages
        assert main_window.pages.widget(0) is main_window.dashboard_page
        assert main_window.pages.widget(17) is main_window.settings_page
        main_window.go_to("pkpd")
        assert main_window.pages.currentIndex() == 3
        assert main_window.pages.currentWidget() is main_window.pkpd_page


class TestHistoryAndNavigation:
    def test_home_back_roundtrip(self, main_window) -> None:
        main_window.go_to("models")
        assert main_window.mode.currentIndex() == 1
        main_window.go_home()
        assert main_window.mode.currentIndex() == 0
        main_window.go_back()
        # back lands on the workspace we came from
        assert main_window.mode.currentIndex() == 1
        assert main_window.workspaces["machine-learning"].currentWidget() is main_window.model_page

    def test_back_disabled_on_fresh_home(self, main_window) -> None:
        assert main_window.top_bar.back_btn.isEnabled() is False

    def test_open_workspace_shows_its_tabs(self, main_window) -> None:
        main_window.open_workspace("pkpd")
        assert main_window.mode.currentIndex() == 1
        assert main_window._workspace_stack.currentWidget() is main_window.workspaces["pkpd"]
        assert main_window.workspace_nav.current() == "pkpd"
        main_window.go_back()
        assert main_window.mode.currentIndex() == 0

    def test_bar_click_routes_to_the_workspace(self, main_window) -> None:
        main_window.workspace_nav.button("generative-ai").click()
        assert main_window.mode.currentIndex() == 1
        assert main_window._workspace_stack.currentWidget() is main_window.workspaces["generative-ai"]
        assert main_window.workspace_nav.current() == "generative-ai"

    def test_introduction_entry_buttons_route(self, main_window) -> None:
        # §12.2: a clear entry point to each workspace
        for card_id, button in [
            (c.card_id, btn) for c in NINE_CARDS
            for btn in main_window.introduction.findChildren(QPushButton)
            if btn.objectName() == f"introductionEntry-{c.card_id}"
        ]:
            button.click()
            assert main_window.mode.currentIndex() == 1, card_id
            assert main_window._workspace_stack.currentWidget() is main_window.workspaces[card_id]
            main_window.go_home()

    def test_unknown_sections_and_workspaces_are_ignored(self, main_window) -> None:
        main_window.go_to("no-such-section")
        assert main_window.mode.currentIndex() == 0
        main_window.open_workspace("not-a-workspace")
        assert main_window.mode.currentIndex() == 0


class TestAdvancedNavigation:
    def test_toggle_shows_and_hides_the_sidebar(self, main_window) -> None:
        main_window.top_bar.advanced_btn.setChecked(True)
        assert main_window.nav.isHidden() is False
        assert main_window.nav.width() >= 190
        main_window.top_bar.advanced_btn.setChecked(False)
        assert main_window.nav.isHidden() is True

    def test_settings_reachable_from_the_top_bar(self, main_window) -> None:
        main_window.top_bar.settings_btn.click()
        assert main_window.mode.currentIndex() == 1
        assert main_window.workspaces["introduction"].currentWidget() is main_window.settings_page

    def test_top_bar_reports_the_current_location(self, main_window) -> None:
        assert main_window.top_bar.location.text() == "Introduction"
        main_window.open_workspace("visualization")
        assert "Visualization" in main_window.top_bar.location.text()
        main_window.go_to("data")
        assert "Data Explorer" in main_window.top_bar.location.text()
        main_window.go_home()


class TestKeyboardAccess:
    def test_alt_home_and_alt_left_registered(self, main_window) -> None:
        shortcuts = [a.shortcut().toString() for a in main_window.actions() if a.shortcut()]
        assert "Alt+Home" in shortcuts and "Alt+Left" in shortcuts

    def test_ctrl_shortcuts_target_the_workspaces(self, main_window) -> None:
        # §12.1: Ctrl+1..9 jump to the nine workspaces in order
        actions = {a.shortcut().toString(): a for a in main_window.actions() if a.shortcut()}
        for i, key in enumerate(DIRECTIVE_ORDER, start=1):
            assert f"Ctrl+{i}" in actions
            actions[f"Ctrl+{i}"].trigger()
            assert main_window.workspace_nav.current() == key, key
        main_window.go_home()


class TestRestoreBehaviour:
    def test_restore_last_is_opt_in_and_default_off(self, main_window, monkeypatch, tmp_path) -> None:
        # §12.2: without the explicit flag the app opens on the
        # Introduction page even with a saved section.
        from PySide6.QtCore import QSettings

        class _Facade:
            def __init__(self, ini):
                self._s = QSettings(str(ini), QSettings.Format.IniFormat)

            def save_geometry(self, g, st=None):
                self._s.setValue("window/geometry", g)
                self._s.sync()

            def geometry(self):
                return None

            def state(self):
                return None

            def save_nav(self, section):
                self._s.setValue("nav/section", section)
                self._s.sync()

            def nav(self):
                return self._s.value("nav/section") or None

            def settings(self):
                return self._s

            def clear(self):
                self._s.clear()

        import ui.main_window as mw

        ini = tmp_path / "restore.ini"
        monkeypatch.setattr(mw, "AppSettings", lambda: _Facade(ini))

        first = mw.MainWindow()
        first.go_to("models")
        first.close()

        second = mw.MainWindow()
        # saved section exists but the restore flag is unset → Introduction
        assert second.settings.nav() == "models"
        assert second.mode.currentIndex() == 0
        second.close()

        # opt-in: the flag restores the last section (into its workspace)
        second2 = mw.MainWindow()
        second2.settings.settings().setValue("nav/restore_last", "true")
        second2.go_to("data")
        second2.close()

        third = mw.MainWindow()
        assert third.mode.currentIndex() == 1
        assert third.workspaces["datasets"].currentWidget() is third.data_page
        third.close()

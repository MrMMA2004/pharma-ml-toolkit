"""The nine-card-first shell (part6 Phase 1, §2.1 + §14.1).

These pins enforce the inversion itself: the app opens on the
nine-card home, the 18-item sidebar is NOT the dominant default
interface (it rides only behind the Advanced-navigation toggle),
Back/Home carry a real history, every specialized page stays
reachable, the nine workspaces exist in the directive's exact order,
and the promoted home grid carries the same §11 live-evidence
contract the Dashboard grid always had.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from ui.main_window import SECTIONS
from ui.nine_cards import NINE_CARDS

DIRECTIVE_ORDER = [
    "introduction", "datasets", "machine-learning", "deep-learning",
    "generative-ai", "molecular-docking", "pkpd",
    "molecular-dynamics-analysis", "visualization",
]


class TestNineCardFirstDefaults:
    def test_opens_on_the_nine_card_home(self, main_window) -> None:
        assert main_window.mode.currentIndex() == 0
        assert main_window.mode.currentWidget() is main_window.home

    def test_sidebar_not_the_dominant_default(self, main_window) -> None:
        # §2.1: hidden at launch, only the Advanced toggle re-shows it;
        # a hidden splitter child is fully excluded from the layout.
        assert main_window.nav.isHidden() is True
        assert not main_window.splitter.sizes() or main_window.splitter.sizes()[0] == 0

    def test_all_eighteen_pages_survive(self, main_window) -> None:
        assert main_window.pages.count() == 18

    def test_nine_workspaces_in_the_directive_order(self, main_window) -> None:
        assert [c.card_id for c in NINE_CARDS] == DIRECTIVE_ORDER
        assert list(main_window.workspaces) == DIRECTIVE_ORDER
        for i, card_id in enumerate(DIRECTIVE_ORDER):
            assert main_window._workspace_stack.widget(i) is main_window.workspaces[card_id]

    def test_no_other_card_exists(self, main_window) -> None:
        # §2.3: no Other card in Phase 1 — no capability was assigned to
        # one yet; the registry decision comes with Phase 2's audit.
        assert len(main_window.workspaces) == 9


class TestHistoryAndNavigation:
    def test_home_back_roundtrip(self, main_window) -> None:
        main_window.go_to("models")
        assert main_window.mode.currentIndex() == 2
        main_window.go_home()
        assert main_window.mode.currentIndex() == 0
        main_window.go_back()
        # back lands on the page we came from, with its section selected
        assert main_window.mode.currentIndex() == 2
        assert main_window.pages.currentWidget() is main_window.model_page

    def test_back_disabled_on_fresh_home(self, main_window) -> None:
        assert main_window.top_bar.back_btn.isEnabled() is False

    def test_open_card_shows_its_workspace(self, main_window) -> None:
        main_window.open_card("pkpd")
        assert main_window.mode.currentIndex() == 1
        ws = main_window._workspace_stack.currentWidget()
        assert ws is main_window.workspaces["pkpd"]
        assert ws.card.title == "PK/PD"
        main_window.go_back()
        assert main_window.mode.currentIndex() == 0

    def test_every_section_reaches_its_page(self, main_window) -> None:
        page_of = {key: page for key, page in zip(
            dict(SECTIONS).keys(),
            [main_window.dashboard_page, main_window.data_page,
             main_window.chemistry_page, main_window.pkpd_page,
             main_window.biology_page, main_window.docking_page,
             main_window.model_page, main_window.mlworkbench_page,
             main_window.unsupervised_page, main_window.deep_learning_page,
             main_window.generative_page, main_window.foundations_page,
             main_window.bestofml_page, main_window.responseai_page,
             main_window.imagelab_page, main_window.molecular_dynamics_page,
             main_window.utilities_page, main_window.settings_page],
        )}
        for section in dict(SECTIONS):
            main_window.go_to(section)
            assert main_window.mode.currentIndex() == 2
            assert main_window.pages.currentWidget() is page_of[section]
        main_window.go_home()

    def test_workspace_entry_buttons_route_to_real_pages(self, main_window) -> None:
        for card_id, ws in main_window.workspaces.items():
            for btn in ws.entry_buttons():
                btn.click()
                assert main_window.mode.currentIndex() == 2, (
                    f"{card_id}: entry {btn.objectName()} did not reach a page"
                )
                main_window.open_card(card_id)

    def test_card_click_opens_the_workspace(self, main_window) -> None:
        # the promoted grid's own signal (part6 §5: the card is the entry)
        main_window.home.card_activated.emit("generative-ai")
        assert main_window.mode.currentIndex() == 1
        assert main_window._workspace_stack.currentWidget() is main_window.workspaces["generative-ai"]

    def test_unknown_sections_and_cards_are_ignored(self, main_window) -> None:
        main_window.go_to("no-such-section")
        assert main_window.mode.currentIndex() == 0
        main_window.open_card("not-a-card")
        assert main_window.mode.currentIndex() == 0


class TestAdvancedNavigation:
    def test_toggle_shows_and_hides_the_sidebar(self, main_window) -> None:
        main_window.top_bar.advanced_btn.setChecked(True)
        assert main_window.nav.isHidden() is False
        assert main_window.nav.width() >= 190  # Navigation's own readable floor
        main_window.top_bar.advanced_btn.setChecked(False)
        assert main_window.nav.isHidden() is True

    def test_settings_reachable_from_the_top_bar(self, main_window) -> None:
        main_window.top_bar.settings_btn.click()
        assert main_window.mode.currentIndex() == 2
        assert main_window.pages.currentWidget() is main_window.settings_page

    def test_top_bar_reports_the_current_location(self, main_window) -> None:
        assert main_window.top_bar.location.text() == "Nine-card home"
        main_window.open_card("visualization")
        assert "Visualization" in main_window.top_bar.location.text()
        main_window.go_to("data")
        assert "Data Explorer" in main_window.top_bar.location.text()
        main_window.go_home()


class TestEvidenceAndShortcuts:
    def test_home_grid_carries_the_live_context(self, main_window) -> None:
        grid = main_window.home.cards_host
        previews = grid.preview_labels()
        assert set(previews) == {c.card_id for c in NINE_CARDS}
        assert previews["datasets"].text().startswith("No dataset loaded")

    def test_alt_home_and_alt_left_registered(self, main_window) -> None:
        shortcuts = [a.shortcut().toString() for a in main_window.actions() if a.shortcut()]
        assert "Alt+Home" in shortcuts and "Alt+Left" in shortcuts

    def test_ctrl_shortcuts_still_route_through_go_to(self, main_window) -> None:
        shortcuts = [a.shortcut().toString() for a in main_window.actions() if a.shortcut()]
        assert "Ctrl+1" in shortcuts and "Ctrl+4" in shortcuts

    def test_workspace_filter_hides_non_matching_entries(self, main_window) -> None:
        main_window.open_card("machine-learning")
        ws = main_window.workspaces["machine-learning"]
        visible_all = [b for b in ws.entry_buttons() if not b.isHidden()]
        assert len(visible_all) == len(ws.entry_buttons())
        ws._filter.setText("unsupervised")
        visible = [b for b in ws.entry_buttons() if not b.isHidden()]
        assert 0 < len(visible) < len(ws.entry_buttons())
        ws._filter.setText("")
        assert len([b for b in ws.entry_buttons() if not b.isHidden()]) == len(ws.entry_buttons())
        main_window.go_home()

    def test_home_grid_buttons_still_route(self, main_window) -> None:
        # the promoted grid's entry buttons keep the part-4 routing (the
        # ML card's first entry is its primary action button)
        grid = main_window.home.cards_host
        btn = next(
            b for b in grid.findChildren(QPushButton)
            if b.objectName() in ("nineCardEntry-models", "nineCardEntry-primary")
            and b.text() == "Open Model Analysis"
        )
        btn.click()
        assert main_window.pages.currentWidget() is main_window.model_page
        main_window.go_home()


class TestRestoreBehaviour:
    def test_restore_last_is_opt_in_and_default_off(self, main_window, monkeypatch, tmp_path) -> None:
        # §2.1: restoring the last workspace must never make the sidebar
        # architecture the default again — without the explicit flag the
        # app opens on the home screen even with a saved section.
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
        # saved section exists but the restore flag is unset → home wins
        assert second.settings.nav() == "models"
        assert second.mode.currentIndex() == 0
        second.close()

        # opt-in: the flag restores the last section
        second2 = mw.MainWindow()
        second2.settings.settings().setValue("nav/restore_last", "true")
        second2.go_to("data")
        second2.close()

        third = mw.MainWindow()
        assert third.mode.currentIndex() == 2
        assert third.pages.currentWidget() is third.data_page
        third.close()

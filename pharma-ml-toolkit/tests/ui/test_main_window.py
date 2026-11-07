"""Main window shell tests: navigation, pages, status bar, persistence."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt


def test_window_constructs_with_all_pages(main_window) -> None:
    assert main_window.pages.count() == 13
    assert main_window.windowTitle() == "PharmaML Studio"


def test_navigation_switches_pages(main_window) -> None:
    order = [
        "dashboard", "data", "chemistry", "pkpd", "biology", "docking",
        "models", "unsupervised", "deep_learning", "generative",
        "foundations", "bestofml", "settings",
    ]
    for i, key in enumerate(order):
        main_window.go_to(key)
        assert main_window.pages.currentIndex() == i
    main_window.go_to("pkpd")
    assert main_window.pages.currentIndex() == 3
    main_window.go_to("biology")
    assert main_window.pages.currentIndex() == 4
    main_window.go_to("docking")
    assert main_window.pages.currentIndex() == 5


def test_nav_emits_section_keys(main_window) -> None:
    seen: list[str] = []
    main_window.nav.section_selected.connect(seen.append)
    main_window.nav.setCurrentRow(2)
    assert seen == ["chemistry"]


def test_unknown_section_is_ignored(main_window) -> None:
    main_window.go_to("dashboard")
    main_window.go_to("no-such-section")
    assert main_window.pages.currentIndex() == 0


def test_status_bar_reflects_job_lifecycle(main_window) -> None:
    main_window.runner.job_started.emit("demo job")
    assert "demo job" in main_window._status.currentMessage()
    main_window.runner.job_ended.emit()
    assert main_window._status.currentMessage() == "Ready"


def test_geometry_persists(qtbot, studio_app, tmp_path, monkeypatch) -> None:
    ini = tmp_path / "geo.ini"
    monkeypatch.setattr(
        "ui.main_window.AppSettings",
        lambda: _SettingsFacade(ini),
    )
    from ui.main_window import MainWindow

    first = MainWindow()
    first.show()
    first.resize(1010, 660)
    qtbot.wait(80)
    first.go_to("models")
    first.close()

    second = MainWindow()
    second.show()
    qtbot.wait(80)
    # The offscreen platform's screen is 800x800, so restoreGeometry clamps
    # the 1010px width to the screen and the window settles on its 860px
    # minimum; the HEIGHT (660, fits the screen) proves the round-trip, as
    # does the persisted nav section. On a real desktop the full geometry
    # restores exactly.
    assert second.height() == 660
    assert second.settings.nav() == "models"
    assert second.settings.geometry() is not None
    second.close()


class _SettingsFacade:
    """Tiny QSettings-backed facade with the AppSettings surface."""

    def __init__(self, ini) -> None:
        self._s = QSettings(str(ini), QSettings.Format.IniFormat)

    def save_geometry(self, geometry, state=None) -> None:
        self._s.setValue("window/geometry", geometry)
        if state is not None:
            self._s.setValue("window/state", state)
        self._s.sync()

    def geometry(self):
        return self._s.value("window/geometry")

    def state(self):
        return None

    def save_nav(self, section: str) -> None:
        self._s.setValue("nav/section", section)
        self._s.sync()

    def nav(self):
        return self._s.value("nav/section") or None

    def settings(self) -> QSettings:
        return self._s

    def clear(self) -> None:
        self._s.clear()


def test_grab_screenshot_nonempty(main_window) -> None:
    pixmap = main_window.grab()
    assert not pixmap.isNull()
    assert pixmap.width() == main_window.width() and pixmap.height() == main_window.height()


def test_ctrl_shortcuts_registered(main_window) -> None:
    actions = [a.shortcut().toString() for a in main_window.actions() if a.shortcut()]
    assert "Ctrl+1" in actions and "Ctrl+4" in actions
    _ = Qt  # keep import honest

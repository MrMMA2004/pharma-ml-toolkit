"""Round 232 -- the platform-aware headless probe (the Windows fix).

The user reported ``studio.py`` launching with console noise but NO
window on Windows + miniconda py312 + PySide6.  Root cause: the old
``ensure_headless()`` forced ``QT_QPA_PLATFORM=offscreen`` whenever
``DISPLAY`` was unset -- and ``DISPLAY`` NEVER exists on Windows (it is
an X11 variable), so every Windows desktop got an invisible window and
a forever-blocking ``app.exec()``.  Wayland-only desktops
(``WAYLAND_DISPLAY`` set, ``DISPLAY`` unset) had the same bug.

These tests pin the decision matrix.  They manipulate only environment
variables and the module's memo -- no Qt objects are created, so the
already-running offscreen QApplication of the UI chunk is unaffected.
"""

from __future__ import annotations

import os
import sys

import pytest
from ui import headless


@pytest.fixture()
def fresh_probe(monkeypatch: pytest.MonkeyPatch):
    """Reset the memo and all platform-relevant env per test."""
    monkeypatch.setattr(headless, "_applied", None)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    yield monkeypatch


def _platform_env() -> str | None:
    return os.environ.get("QT_QPA_PLATFORM")


class TestOffscreenSelection:
    """Linux-class, displayless boxes still get the offscreen plugin."""

    def test_linux_no_display_goes_offscreen(self, fresh_probe):
        fresh_probe.setattr(sys, "platform", "linux")
        applied = headless.ensure_headless()
        assert _platform_env() == "offscreen"
        assert applied["platform"] == "offscreen"

    def test_linux_wayland_only_stays_native(self, fresh_probe):
        """Wayland desktop: WAYLAND_DISPLAY without DISPLAY must not
        be mistaken for a headless box (the round-232 sibling bug)."""
        fresh_probe.setattr(sys, "platform", "linux")
        fresh_probe.setenv("WAYLAND_DISPLAY", "wayland-0")
        applied = headless.ensure_headless()
        assert _platform_env() is None  # Qt picks its native plugin
        assert applied["platform"] == "native"

    def test_linux_x11_stays_native(self, fresh_probe):
        fresh_probe.setattr(sys, "platform", "linux")
        fresh_probe.setenv("DISPLAY", ":0")
        applied = headless.ensure_headless()
        assert _platform_env() is None
        assert applied["platform"] == "native"


class TestDesktopPlatformsNeverForcedOffscreen:
    """The user-facing regression: Windows and macOS desktops."""

    def test_windows_keeps_native_platform(self, fresh_probe):
        """Windows has no DISPLAY variable -- the old probe forced
        offscreen here and the GUI never appeared (round 232 fix)."""
        fresh_probe.setattr(sys, "platform", "win32")
        applied = headless.ensure_headless()
        assert _platform_env() is None
        assert applied["platform"] == "native"

    def test_macos_keeps_native_platform(self, fresh_probe):
        fresh_probe.setattr(sys, "platform", "darwin")
        applied = headless.ensure_headless()
        assert _platform_env() is None
        assert applied["platform"] == "native"


class TestExplicitChoiceRespected:
    """A caller-set QT_QPA_PLATFORM always wins."""

    def test_explicit_platform_untouched(self, fresh_probe):
        fresh_probe.setattr(sys, "platform", "linux")
        fresh_probe.setenv("QT_QPA_PLATFORM", "minimal")
        applied = headless.ensure_headless()
        assert _platform_env() == "minimal"
        assert applied["platform"] is None  # we did not choose


class TestIdempotence:
    """Second call is a memo hit and re-applies nothing."""

    def test_second_call_returns_memo(self, fresh_probe):
        fresh_probe.setattr(sys, "platform", "linux")
        headless.ensure_headless()  # first call applies; the memo is asserted below
        assert _platform_env() == "offscreen"
        headless._applied["platform"] = "already-run"
        second = headless.ensure_headless()
        assert second["platform"] == "already-run"
        assert _platform_env() == "offscreen"  # env untouched by memo hit

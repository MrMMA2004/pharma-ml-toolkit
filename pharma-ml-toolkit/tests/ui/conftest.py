"""Test bootstrap for the Studio UI tests.

``ensure_headless()`` runs before ANY PySide6 import: on this sandbox
it preloads the committed EGL no-op stub (``ui/native/libEGL.so.1``,
RTLD_GLOBAL, real EGL probed first) and selects the offscreen platform
when no DISPLAY exists. pytest-qt supplies the ``qapp`` fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TK))  # ui/ + studio.py live at the toolkit root

from ui.headless import ensure_headless

ensure_headless()

import pytest


@pytest.fixture()
def studio_app(qapp):
    """The shared QApplication with a clean native (unstyled) base."""
    qapp.setStyleSheet("")
    yield qapp


@pytest.fixture()
def theme_manager(studio_app):
    from ui.theme_manager import ThemeManager

    return ThemeManager(studio_app)


@pytest.fixture()
def main_window(studio_app):
    from ui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()

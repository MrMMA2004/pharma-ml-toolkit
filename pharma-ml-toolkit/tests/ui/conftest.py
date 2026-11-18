"""Test bootstrap for the Studio UI tests.

``ensure_headless()`` runs before ANY PySide6 import: on this sandbox
it preloads the committed EGL no-op stub (``ui/native/libEGL.so.1``,
RTLD_GLOBAL, real EGL probed first) and selects the offscreen platform
when no DISPLAY exists. pytest-qt supplies the ``qapp`` fixture.
"""

from __future__ import annotations

import gc
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
    # PySide6 6.12: setStyleSheet lives on QApplication (widgets),
    # not QGuiApplication — a bare QGuiApplication left behind by an
    # earlier suite (e.g. the qt-doctor subprocess checks) would
    # raise here under 6.12 while working under 6.8.
    if hasattr(qapp, "setStyleSheet"):
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
    # (r271 gate-find, fixed at the source) close() only HIDES the
    # window: the C++ widget tree (18 subpages, pyqtgraph items,
    # molecule pixmaps) stayed alive with its Python wrappers, so a
    # 23-test suite like test_nine_card_first retained ~150 MiB per
    # instantiation and OOM'd the 4 GiB sandbox at test ~21 (SIGKILL,
    # twice, deterministically).  deleteLater + a processEvents pump
    # actually destroys the tree; gc.collect() then releases the
    # Python-side cycles.  Every UI suite shares this fixture, so the
    # whole gate's UI half gets the bounded-memory teardown for free.
    window.close()
    window.deleteLater()
    studio_app.processEvents()
    gc.collect()

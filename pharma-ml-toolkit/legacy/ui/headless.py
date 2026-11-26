"""Headless/environment bootstrap for PharmaML Studio (PySide6 GUI).

Import this BEFORE any PySide6 import when the app or the tests run on
a machine that may lack a display or system GL libraries (CI sandboxes,
containers). It is a no-op on a normal desktop:

1. EGL fallback (round 206): PySide6's ``QtGui`` links ``libEGL.so.1``
   at load time even for the offscreen platform plugin (software
   rasterization, no real GL context). Sandboxes that ship ``libGL``
   but not ``libEGL`` fail at ``import PySide6.QtWidgets`` with
   "libEGL.so.1: cannot open shared object file". We first probe for a
   real EGL; only when that is missing do we preload the committed
   no-op stub ``ui/native/libEGL.so.1`` with ``RTLD_GLOBAL`` so its
   symbols satisfy the dynamic linker. The offscreen plugin never
   actually issues GL calls through the stub entries.

2. Platform plugin: with no display server reachable (and not already
   configured) we select Qt's ``offscreen`` platform so the app and the
   test suite run truly headless. A real desktop keeps its native
   platform.  Round 232 (the Windows black-screen fix): the probe is now
   platform-aware -- ``DISPLAY`` only exists on X11, so the old test
   ``not os.environ.get("DISPLAY")`` was TRUE on every Windows and
   Wayland desktop too, silently forcing ``offscreen`` there: the app
   ran with an invisible window (``app.exec()`` blocked, no GUI ever
   appeared -- the user-reported "studio.py opens no window").  Now
   offscreen is only chosen on Linux-class platforms that expose NEITHER
   ``DISPLAY`` nor ``WAYLAND_DISPLAY``; Windows/macOS always keep their
   native platform plugin.

3. Binding pin: ``PYQTGRAPH_QT_LIB=PySide6`` so pyqtgraph 0.13 binds to
   PySide6 explicitly (it would otherwise probe Qt bindings via qtpy).

The function is idempotent: calling it twice changes nothing the second
time, and it returns the environment dict it applied for observability.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_STUB = _NATIVE_DIR / "libEGL.so.1"

_applied: dict[str, str | None] | None = None


def _real_egl_available() -> bool:
    """True when a system libEGL.so.1 resolves and loads."""
    try:
        ctypes.CDLL("libEGL.so.1")
    except OSError:
        return False
    return True


def _load_stub() -> bool:
    """Preload the committed EGL no-op stub with RTLD_GLOBAL."""
    if not _STUB.exists():  # pragma: no cover - binary missing (e.g. exotic arch)
        return False
    try:
        ctypes.CDLL(str(_STUB), mode=ctypes.RTLD_GLOBAL)
    except OSError:  # pragma: no cover - wrong architecture / corrupt file
        return False
    return True


def ensure_headless() -> dict[str, str | None]:
    """Prepare the process environment for Qt (idempotent).

    Returns a dict of what was applied: keys ``egl`` (``"real"`` /
    ``"stub"`` / ``None``) and ``platform`` (the QT_QPA_PLATFORM that is
    now in force, if we set it).
    """
    global _applied
    if _applied is not None:
        return _applied

    applied: dict[str, str | None] = {"egl": None, "platform": None}

    # 1. EGL probe / stub preload (Linux only; macOS/Windows ship their own).
    if sys.platform.startswith("linux") and not _real_egl_available():
        applied["egl"] = "stub" if _load_stub() else "unavailable"

    # 2. Platform plugin: pick offscreen only on Linux-class platforms
    #    with NO display server in reach.  ``DISPLAY`` is an X11 variable;
    #    Windows and macOS never set it, and a Wayland desktop may set
    #    only ``WAYLAND_DISPLAY`` -- so the display probe itself must be
    #    platform-aware or the GUI would be forced offscreen (invisible)
    #    on exactly the desktops users launch it from (round 232).
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if (
        sys.platform.startswith("linux")
        and not os.environ.get("QT_QPA_PLATFORM")
        and not _has_display
    ):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        applied["platform"] = "offscreen"
    elif not os.environ.get("QT_QPA_PLATFORM"):
        # Desktop platforms (``win32`` / ``darwin`` / Linux-with-display):
        # record what the native plugin will be, for observability.
        applied["platform"] = "native"

    # 3. pyqtgraph binding pin (PySide6 is the only binding we ship).
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    _applied = applied
    return applied

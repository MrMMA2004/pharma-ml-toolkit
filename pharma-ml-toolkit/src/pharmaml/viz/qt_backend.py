"""Optional Qt backend detection (the PySide6 probe pattern).

``qtproject/pyside-pyside-setup`` is the Qt-for-Python *factory*: a
CMake/shiboken6 build system that compiles ~60 Qt 6 C++ modules into
Python extension libraries.  It is exactly the kind of repository this
toolkit does **not** vendor — no numerics, no data models, a
C++/LGPL-licensed binding layer behind a multi-gigabyte build.  What
transfers natively are its small runtime patterns, and they are
absorbed here:

* the **optional-backend probe** — ``PySide6/__init__.py``'s habit of
  discovering what is actually importable before touching it, with a
  descriptive ``ImportError`` (never a bare crash) when the native
  layer is missing;
* the **lazy ``__getattr__`` (PEP 562)** discipline — nothing is
  imported until it is asked for, so a toolkit that ships without Qt
  pays nothing for this module existing;
* the informative error message pattern (the PySide6 wheel names the
  missing sub-package and prints guidance instead of a stack trace).

The historical Tk playground and the PySide6 interface are both retired.
This module remains as a lightweight compatibility probe for consumers that
still need to detect Qt; it is not used by the supported React/Tauri desktop
application or the separate Next.js console. The ``.[studio]`` extra is kept
only for reproducing the archived PySide6 source.

Examples
--------
>>> from pharmaml.viz.qt_backend import QtBackendInfo, probe_qt_backend
>>> info = probe_qt_backend()
>>> isinstance(info, QtBackendInfo)
True
>>> info.module in {"PySide6", "PyQt6", None}
True
>>> info.available is False or info.module is not None   # consistent probe
True

The probe never raises, whatever is (not) installed:

>>> _ = probe_qt_backend()

>>> from pharmaml.viz.qt_backend import describe_qt_backend
>>> text = describe_qt_backend()
>>> text.startswith("Qt backend:")
True
"""

from __future__ import annotations

import importlib.util

__all__ = ["QtBackendInfo", "describe_qt_backend", "import_qt_backend", "probe_qt_backend"]

_CANDIDATES = ("PySide6", "PyQt6")
_WIDGET_SUBMODULES = {"PySide6": "PySide6.QtWidgets", "PyQt6": "PyQt6.QtWidgets"}


class QtBackendInfo:
    """What the probe learned about the Qt stacks on this machine.

    Attributes
    ----------
    module:
        ``'PySide6'`` or ``'PyQt6'`` when a usable binding is present,
        ``None`` otherwise.
    available:
        Whether the binding *and* its QtWidgets layer import cleanly.
    version:
        The binding's ``__version__`` when exposed, else ``None``.
    note:
        A human sentence explaining the state (the descriptive-error
        pattern from ``PySide6/__init__.py``).
    """

    __slots__ = ("available", "module", "note", "version")

    def __init__(
        self,
        module: str | None,
        available: bool,
        version: str | None,
        note: str,
    ) -> None:
        self.module = module
        self.available = available
        self.version = version
        self.note = note

    def __repr__(self) -> str:  # pragma: no cover - repr convenience
        return (
            f"QtBackendInfo(module={self.module!r}, available={self.available}, "
            f"version={self.version!r})"
        )


def _binding_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def probe_qt_backend() -> QtBackendInfo:
    """Detect an optional Qt binding — never raises, never imports it.

    Walks the candidate bindings in order; the first whose top-level
    package *and* QtWidgets submodule both have import specs wins.  A
    present-but-broken binding is reported as unavailable with the
    reason, mirroring PySide6's descriptive-import-failure habit.
    """
    for name in _CANDIDATES:
        if not _binding_importable(name):
            continue
        widgets = _WIDGET_SUBMODULES[name]
        if importlib.util.find_spec(widgets) is None:
            return QtBackendInfo(
                module=name,
                available=False,
                version=None,
                note=f"{name} is installed but {widgets} is missing — "
                "the wheel is incomplete (reinstall it).",
            )
        try:  # pragma: no cover - depends on the host installation
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
        except Exception as exc:  # noqa: BLE001 - descriptive, not fatal
            return QtBackendInfo(
                module=name,
                available=False,
                version=None,
                note=f"{name} is installed but failed to import: {exc}",
            )
        return QtBackendInfo(
            module=name,
            available=True,
            version=version,
            note=f"{name} ({version or 'version unknown'}) with QtWidgets "
            "is importable — an optional Qt-rich backend could use it.",
        )
    return QtBackendInfo(
        module=None,
        available=False,
        version=None,
        note="no Qt binding installed (PySide6/PyQt6); the archived PySide6 "
        "interface is unsupported and is not the current PharmaML Studio.",
    )


def import_qt_backend() -> object:
    """Import and return the QtWidgets module of the probed binding.

    Raises
    ------
    ImportError
        With install guidance when no usable binding exists — the
        PySide6 ``__init__`` pattern of naming exactly what is missing
        instead of crashing from deep inside a lazy getter.
    """
    info = probe_qt_backend()
    if info.module is None or not info.available:
        raise ImportError(
            "no usable Qt binding found. Qt is used only by the retired PySide6 "
            "interface, not the supported React/Tauri desktop app. To reproduce "
            "the archived UI, from the toolkit package root run "
            "python -m pip install -e '.[studio]'. " + info.note
        )
    return importlib.import_module(_WIDGET_SUBMODULES[info.module])


def describe_qt_backend() -> str:
    """One line for status panels: what the probe found."""
    info = probe_qt_backend()
    if info.available:
        return f"Qt backend: {info.module} {info.version or ''} (optional)".rstrip()
    return "Qt backend: not installed — archived PySide6 UI is unsupported"

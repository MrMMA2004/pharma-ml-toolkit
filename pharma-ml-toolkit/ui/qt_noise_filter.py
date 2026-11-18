"""Console-noise filter for known-benign Qt startup warnings (round 231).

What a Windows user actually sees with a pip/Miniconda PySide6 6.8.x
(reported against v1.2.142, ``python studio.py``)::

    QFontDatabase: Cannot find font directory
        C:/Users/np/miniconda3/envs/py312/Lib/site-packages/PySide6/lib/fonts.
    Note that Qt no longer ships fonts. Deploy some (from
        https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
    qt.core.qobject.connect: QObject::connect(QStyleHints, QStyleHints):
        unique connections require a pointer to member function of a
        QObject subclass
    (the QStyleHints line repeats four times)

Both families are harmless -- verified against the Qt/PySide6 issue
trackers and the Studio's own startup:

1. **The fonts-dir notice** fires once during ``QGuiApplication`` font
   database setup: the PySide6 *wheels* do not ship a ``lib/fonts``
   directory, and on Windows Qt renders text through the native
   GDI/DirectWrite stack, so the missing directory costs nothing.
   Linux (fontconfig) and macOS (CoreText) wheels are unaffected.
   The notice is the QPA plugin's generic advice, not an error Studio
   can act on -- shipping font files inside the toolkit would be wrong.

2. **The QStyleHints connect lines** are internal Qt 6.8 plumbing:
   ``QGuiApplicationPrivate`` wires ``QStyleHints`` property-notify
   signals through the unique-connection checker, which logs when the
   slot is not a plain member-function pointer.  Nothing in PharmaML
   Studio issues these connections; they appear before ``MainWindow``
   exists and change nothing about scaling, fonts, or input.

The filter drops *exactly* these messages and passes everything else
through with the default ``category: message`` formatting, so REAL
warnings and criticals stay visible.  ``PHARMAML_STUDIO_QT_VERBOSE=1``
disables the filter entirely (escape hatch for debugging Qt itself).

Install it before creating the ``QApplication`` -- both warning
families fire during application construction, not afterwards.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "BENIGN_MARKERS",
    "install_qt_message_filter",
    "is_benign_qt_noise",
]

#: Substrings that identify the two known-benign message families.
#: Matched against the message body (the category prefix is not part
#: of the message the handler receives).
BENIGN_MARKERS: tuple[str, ...] = (
    # family 1: missing wheel fonts dir (Windows GDI/DirectWrite fallback)
    "Cannot find font directory",
    "Note that Qt no longer ships fonts",
    # family 2: Qt 6.8 QStyleHints internal unique-connection chatter
    "QObject::connect(QStyleHints, QStyleHints)",
)

_VERBOSE_ENV = "PHARMAML_STUDIO_QT_VERBOSE"


def is_benign_qt_noise(message: str) -> bool:
    """True when ``message`` is one of the known-benign startup notices.

    The check is deliberately substring-exact against the two families
    documented in the module docstring -- it must never catch a real
    warning (a real font failure or a real connect misuse would not
    carry these exact signatures).
    """
    return any(marker in message for marker in BENIGN_MARKERS)


def _category_of(context: object) -> str:
    """Extract the logging category from a Qt message context.

    PySide6 delivers a ``QMessageLogContext`` with a ``.category``
    string; guard the exotic shapes (``None``, tuple) so the handler
    can never raise from inside Qt's logging path.
    """
    category = getattr(context, "category", None)
    if isinstance(category, str):
        return category
    try:  # pragma: no cover; defensive context shapes
        return str(context[3])  # type: ignore[index]
    except Exception:  # noqa: BLE001 - pragma: no cover
        return "default"


def _format_passthrough(msg_type: object, context: object, message: str) -> str:
    """Render a passing message the way Qt's default handler does."""
    category = _category_of(context)
    prefix = f"{category}: " if category and category != "default" else ""
    severity = ""
    try:
        from PySide6.QtCore import QtMsgType

        if msg_type == QtMsgType.QtCriticalMsg:
            severity = "Critical: "
        elif msg_type == QtMsgType.QtFatalMsg:
            severity = "Fatal: "
        elif msg_type == QtMsgType.QtInfoMsg:
            severity = "Info: "
    except Exception:  # noqa: BLE001 - pragma: no cover; import never fails
        severity = ""
    return f"{severity}{prefix}{message}"


def install_qt_message_filter() -> bool:
    """Install the benign-noise message handler.

    Returns ``True`` when the filter is now active, ``False`` when the
    verbose escape hatch (``PHARMAML_STUDIO_QT_VERBOSE=1``) suppressed
    installation or the Qt binding is unavailable.  Idempotent: a
    second call swaps in the same handler (the previous one is ours).
    """
    if os.environ.get(_VERBOSE_ENV, "").strip() not in ("", "0", "false", "False"):
        return False
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:  # noqa: BLE001 - pragma: no cover; studio imports PySide6 anyway
        return False

    def _handler(msg_type: object, context: object, message: str) -> None:
        if is_benign_qt_noise(message):
            return
        sys.stderr.write(_format_passthrough(msg_type, context, message) + "\n")
        sys.stderr.flush()

    qInstallMessageHandler(_handler)
    return True

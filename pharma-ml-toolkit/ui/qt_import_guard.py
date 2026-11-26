"""Import-failure guidance for the archived PySide6 interface.

The supported desktop application is React + TypeScript hosted by Tauri 2;
the separate Next.js console is another application. The PySide6 modules in
``ui/`` and ``legacy/studio.py`` are retained only for historical
reproducibility and are not supported launch paths. This guard keeps failures
from that archived code understandable without presenting it as the current
PharmaML Studio.
"""

from __future__ import annotations

import sys

__all__ = ["QT_ENV_EXIT_CODE", "qt_import_guidance"]

#: Distinct exit status for an archived Qt-binding failure.
QT_ENV_EXIT_CODE = 2

_RULE = "=" * 72


def _py() -> str:
    """The interpreter that launched the archived UI, quoted for copy-paste."""
    return sys.executable or "python"


def qt_import_guidance(exc: BaseException) -> str:
    """Format a concise warning for failures in the retired PySide6 UI."""
    py = _py()
    msg = str(exc)
    missing = "no module named" in msg.lower() and "pyside6" in msg.lower()
    lines = [
        _RULE,
        "The retired PySide6 interface could not load its Qt binding.",
        "It is archived and unsupported; it is not the current PharmaML Studio.",
        "The supported desktop app is React + TypeScript hosted by Tauri 2.",
        "The Next.js console is a separate web application.",
        "",
        f"Error: {msg}",
        "",
    ]
    if missing:
        lines += [
            "For archival reproduction only, install the legacy extra from the",
            "toolkit package root:",
            f'  "{py}" -m pip install -e ".[studio]"',
            "This extra does not install the supported React/Tauri application.",
        ]
    else:
        lines += [
            "This is a broken Qt environment used by retired source.",
            "No PySide6 reinstall recipe is provided for the unsupported UI.",
            "See frontend/README.md for the supported desktop application's",
            "development instructions, and legacy/README.md for archival notes.",
        ]
    lines += ["", _RULE, ""]
    return "\n".join(lines)

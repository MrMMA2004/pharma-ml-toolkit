"""Friendly failure for broken PySide6 installs (round 237).

What a Windows user actually sees without this guard (reported against
v1.2.147, Miniconda ``py312``, ``python studio.py``)::

    PySide6/__init__.py: Unable to import Shiboken from ...
    Traceback (most recent call last):
      ...
      File "...\\site-packages\\PySide6\\__init__.py", line 140, in <module>
        _setupQtDirectories()
      File "...\\site-packages\\shiboken6\\__init__.py", line 27, in <module>
        from shiboken6.Shiboken import *
    ImportError: DLL load failed while importing Shiboken: The specified
    module could not be found.

That is an environment problem, not a PharmaML Studio bug: the failure
is inside the binding's own import chain, before any Studio code runs
(the usual causes are a half-finished wheel upgrade, a ``shiboken6``
that does not match ``PySide6``, or a missing MSVC runtime).  But the
raw traceback gives the user nothing to act on, so :func:`ui.run`
catches the ``ImportError`` at exactly that import, prints
:func:`qt_import_guidance` -- the diagnosis plus copy-pasteable fix
commands built from ``sys.executable`` so they target the interpreter
that actually launched Studio -- and returns :data:`QT_ENV_EXIT_CODE`
(instead of raising), keeping the console clean and scriptable.

The guidance distinguishes the two failure families it can see:

* ``No module named 'PySide6'`` -- the binding simply is not installed
  in this interpreter; the fix is a plain install.
* anything else (DLL load failed, shiboken mismatch, ...) -- an
  install exists but is broken; the fix is a clean uninstall +
  reinstall of the whole wheel family.  ``PySide6``,
  ``PySide6-Addons``, ``PySide6-Essentials`` and ``shiboken6`` are one
  family and must move as a set -- that is why partial upgrades break
  exactly like the report above.

``python studio.py --doctor`` (round 237's environment self-check,
:mod:`ui.doctor`) is referenced from both branches so users get the
full diagnosis path the next time anything in the stack breaks.
"""

from __future__ import annotations

import sys

__all__ = ["QT_ENV_EXIT_CODE", "qt_import_guidance"]

#: Distinct exit status for "the Qt binding could not be loaded" so
#: launchers/scripts can tell an environment failure from an app
#: failure (0) or a usage error (typically 2 from argparse -- pick
#: this deliberately: Studio has no argparse, so 2 is unambiguous).
QT_ENV_EXIT_CODE = 2

_RULE = "=" * 72


def _py() -> str:
    """The interpreter that launched Studio, quoted for copy-paste."""
    return sys.executable or "python"


def qt_import_guidance(exc: BaseException) -> str:
    """The actionable message printed when the Qt import fails.

    Pure formatter: takes the caught exception, returns the full
    multi-line report (trailing newline included) that :func:`ui.run`
    writes to stderr.  Branches on the two failure families described
    in the module docstring so the fix matches the diagnosis.
    """
    py = _py()
    msg = str(exc)
    missing = "no module named" in msg.lower() and "pyside6" in msg.lower()
    lines = [
        _RULE,
        "PharmaML Studio could not load PySide6 (the Qt binding).",
        "",
        f"Error: {msg}",
        "",
    ]
    if missing:
        lines += [
            "PySide6 is not installed in this interpreter.",
            "",
            "Fix:",
            f'  1) "{py}" -m pip install PySide6',
            f'  2) "{py}" studio.py --doctor    (full environment check)',
        ]
    else:
        lines += [
            "This is an environment problem, not a Studio bug: the",
            "PySide6 install in this interpreter is broken.  The usual",
            "causes: a half-finished upgrade, a shiboken6 that does not",
            "match PySide6, or a missing MSVC runtime.",
            "",
            "Fix -- run BOTH steps with the interpreter that launches Studio:",
            (
                f'  1) "{py}" -m pip uninstall -y'
                " PySide6 PySide6-Addons PySide6-Essentials shiboken6"
            ),
            f'  2) "{py}" -m pip install --upgrade PySide6',
            "",
            "If it still fails:",
            "  - install the Microsoft Visual C++ Redistributable (x64,",
            "    2015-2022) and retry;",
            (
                "  - conda environments: prefer"
                " `conda install -c conda-forge pyside6`"
            ),
            "    and never mix pip and conda PySide6 in one env;",
            f'  - full diagnosis: "{py}" studio.py --doctor',
        ]
    lines += ["", _RULE, ""]
    return "\n".join(lines)

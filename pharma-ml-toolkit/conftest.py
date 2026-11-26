"""Root conftest — EGL-safe pytest for EVERY invocation (round 230).

pytest-qt is installed in the venv, so every pytest run imports
``PySide6.QtGui`` at ``pytest_configure`` — BEFORE any test module (and
before the legacy ``tests_ui/conftest.py`` bootstrap) gets a chance to
preload the EGL stub.  In containers without system ``libEGL.so.1`` (this
sandbox's resting state) that import dies with ``ImportError:
libEGL.so.1`` and the whole session INTERNALERRORs.  Library-only runs
were exposed twice over: their args (``tests/test_a*.py``, the doctest
chunk ``src/pharmaml``) never pull the legacy conftest in as an initial
conftest, so nothing preloads the stub — the legacy GUI runs only
survived because their conftest did.

The root conftest loads for EVERY argument path under the toolkit root,
so this is the one place the preload is unconditional:
``ui.headless.ensure_headless()`` (the round-206 mechanism) probes a
real EGL first and only then preloads the committed no-op stub
``ui/native/libEGL.so.1`` (``RTLD_GLOBAL``) and selects the offscreen
platform when no ``DISPLAY`` exists.  Idempotent, and a no-op on a
normal desktop with real GL libraries.

part18 §10 quarantine completion: the legacy PySide6 GUI tree now lives
at ``legacy/ui/`` (its launcher at ``legacy/studio.py``, the archival
test family at ``legacy/tests_ui/``) — the Qt-free ``pharmaml.studio``
persistence modules stayed in ``src/`` where they legitimately serve the
FastAPI backend.  ``legacy/`` joins the path so the quarantined
``ui.*`` imports (the stress-test GUI lane and the archival tests)
still resolve while nothing under ``tests/`` or ``src/`` gains a Qt
dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parent
if str(_TK) not in sys.path:  # the toolkit root: src/ + contracts/ + docs/
    sys.path.insert(0, str(_TK))
_LEGACY = _TK / "legacy"
if _LEGACY.is_dir() and str(_LEGACY) not in sys.path:
    # the quarantined GUI tree (legacy/ui/) + its archival tests
    sys.path.insert(0, str(_LEGACY))

from ui.headless import ensure_headless

ensure_headless()

"""Root conftest — EGL-safe pytest for EVERY invocation (round 230).

pytest-qt is installed in the venv, so every pytest run imports
``PySide6.QtGui`` at ``pytest_configure`` — BEFORE any test module (and
before the ``tests/ui/conftest.py`` bootstrap) gets a chance to preload
the EGL stub.  In containers without system ``libEGL.so.1`` (this
sandbox's resting state) that import dies with ``ImportError:
libEGL.so.1`` and the whole session INTERNALERRORs.  Library-only runs
were exposed twice over: their args (``tests/test_a*.py``, the doctest
chunk ``src/pharmaml``) never pull ``tests/ui/conftest.py`` in as an
initial conftest, so nothing preloads the stub — the ``tests/ui`` runs
only survived because their conftest did.

The root conftest loads for EVERY argument path under the toolkit root,
so this is the one place the preload is unconditional:
``ui.headless.ensure_headless()`` (the round-206 mechanism) probes a
real EGL first and only then preloads the committed no-op stub
``ui/native/libEGL.so.1`` (``RTLD_GLOBAL``) and selects the offscreen
platform when no ``DISPLAY`` exists.  Idempotent, and a no-op on a
normal desktop with real GL libraries.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parent
if str(_TK) not in sys.path:  # ui/ + studio.py live at the toolkit root
    sys.path.insert(0, str(_TK))

from ui.headless import ensure_headless

ensure_headless()

#!/usr/bin/env python
"""PharmaML Studio -- application entry point (directive #16, increment 1).

Launch with::

    python studio.py             # desktop session
    QT_QPA_PLATFORM=offscreen python studio.py   # headless / CI
    python studio.py --doctor    # environment self-check (round 237)

``--doctor`` reports the whole launch environment (interpreter, the
PySide6/shiboken6 wheel family, the GUI providers, the science stack)
with a fix printed under every failing check, and exits 0/1 -- it
never starts the GUI and never mutates the environment.  A broken
PySide6 install (the classic ``DLL load failed while importing
Shiboken`` on Windows) exits 2 with the reinstall recipe on stderr.

Round 228 (the retirement round): the legacy Tkinter playground
(``playground.py``) is RETIRED — every tab's capability lives here
(see ``docs/gui/GUI_AUDIT.md``'s Part-2 completion assessment) and
this application is THE front end.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import run

if __name__ == "__main__":
    # Frozen-application initialization (the part9 §13 audit): the only
    # spawn-context pool in the app lives in pharmaml/docking/ligands.py
    # (3D structure generation), launched from worker threads; the GUI
    # itself is guarded by this very ``__main__`` guard, so children
    # never re-launch it.  ``freeze_support()`` is a no-op in every
    # unfrozen session (source checkouts, wheels, the test suites);
    # in a PyInstaller-frozen Windows build it makes the spawn
    # children bootstrap correctly instead of falling through the
    # module body.  It is called BEFORE any other work by design.
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    raise SystemExit(run(sys.argv))

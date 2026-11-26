#!/usr/bin/env python
"""RETIRED (part18 §10, the owner's mandatory directive) -- this file
is NO LONGER an active supported application entry point.

The supported application surfaces are:
  * the React/Tauri 2 desktop app (frontend/ + src-tauri/) with the
    frozen FastAPI sidecar (packaging/windows/backend_entry.py);
  * the Python scientific toolkit + CLI (src/pharmaml/, __main__.py);
  * the FastAPI backend (src/pharmaml/api/).

This launcher lives on ONLY as quarantined archival source for
reproducibility (the legacy PySide6 GUI: ui/ pages + workspace.py)
and is NOT part of production packaging (the sidecar excludes the
whole Qt stack) and NOT covered by the product's support.  See
legacy/README.md.

Original header follows:

PharmaML Studio -- application entry point (directive #16, increment 1).

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
# part18 §10 quarantine completion: the whole legacy GUI tree moved in
# here with the launcher (legacy/ui/ beside this file), so THIS
# directory on sys.path is what makes ``from ui import run`` and the
# package imports resolve — nothing outside legacy/ is touched.

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

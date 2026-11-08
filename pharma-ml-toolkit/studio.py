#!/usr/bin/env python
"""PharmaML Studio -- application entry point (directive #16, increment 1).

Launch with::

    python studio.py             # desktop session
    QT_QPA_PLATFORM=offscreen python studio.py   # headless / CI

Round 228 (the retirement round): the legacy Tkinter playground
(``playground.py``) is RETIRED — every tab's capability lives here
(see ``docs/gui/GUI_AUDIT.md``'s Part-2 completion assessment) and
this application is THE front end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import run

if __name__ == "__main__":
    raise SystemExit(run(sys.argv))

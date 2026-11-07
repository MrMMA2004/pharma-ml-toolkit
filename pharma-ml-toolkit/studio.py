#!/usr/bin/env python
"""PharmaML Studio -- application entry point (directive #16, increment 1).

Launch with::

    python studio.py             # desktop session
    QT_QPA_PLATFORM=offscreen python studio.py   # headless / CI

The Tkinter playground (``playground.py``) remains fully functional
alongside this application; sections migrate here incrementally and
are only retired there after their replacements are verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import run

if __name__ == "__main__":
    raise SystemExit(run(sys.argv))

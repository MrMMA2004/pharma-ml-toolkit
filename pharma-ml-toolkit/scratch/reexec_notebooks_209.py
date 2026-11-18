#!/usr/bin/env python3
"""round-260 §4.3 forwarder: the canonical notebook re-execution ritual
lives in ``tools/reexec_notebooks.py`` now.  This shim keeps the
historical ``scratch/reexec_notebooks_209.py`` path alive (the cron
ritual references it); it must still run IN THE FOREGROUND -- detached
launches die silently in this sandbox (the round-209 lesson).
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "reexec_notebooks.py"),
    run_name="__main__",
)

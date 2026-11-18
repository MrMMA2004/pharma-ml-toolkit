#!/usr/bin/env python3
"""round-260 §4.3 forwarder: the stale seven-chunk reference.  The
master job's own text still says ``scratch/run_suite_chunks_225.py``
(the round-225 name) while the ritual has been the NINE-chunk gate
since round 250 -- this shim lands that stale reference on the
CANONICAL runner instead of the retired chunking.
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "run_suite_chunks.py"),
    run_name="__main__",
)

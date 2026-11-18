#!/usr/bin/env python3
"""round-260 §4.3 forwarder: the canonical NINE-chunk gate runner lives
in ``tools/run_suite_chunks.py`` now.  This shim keeps the historical
``scratch/run_suite_chunks_250.py`` path alive; the sentinel contract
is unchanged (``SUITE250 DONE: N failing chunk(s)`` in
``/tmp/suite250.txt``) because the canonical script is the round-250
recipe verbatim.
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "run_suite_chunks.py"),
    run_name="__main__",
)

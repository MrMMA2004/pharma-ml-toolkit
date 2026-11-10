#!/usr/bin/env python3
"""round-260 §4.3 forwarder: the canonical four-home builder lives in
``tools/build_zip.py`` now (MASTER_PROMPT part4: canonical build
utilities move out of the research-history scratch/).  This shim keeps
the historical invocation path alive -- the round ritual and muscle
memory say ``scratch/build_zip_210.py`` -- with zero content
duplication: it exec's the canonical script under ``__main__``.
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "build_zip.py"),
    run_name="__main__",
)

#!/usr/bin/env python3
"""Checkout compatibility entry point; implementation lives in tools/."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "strip_notebook_outputs.py"),
    run_name="__main__",
)

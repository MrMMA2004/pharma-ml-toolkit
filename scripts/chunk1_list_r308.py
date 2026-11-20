#!/usr/bin/env python3
"""Round 308: recompute chunk-1 exactly as scratch/run_suite_chunks_304.py
does (same balance algorithm, same exclusions) and print its file list."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/scratch")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "chunks304", "/home/z/my-project/scratch/run_suite_chunks_304.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TOOLKIT = mod.TOOLKIT
files = sorted(TOOLKIT.rglob("test_*.py"))
nine = mod.NINE_CARD_SUITE
assert nine in files
files.remove(nine)
files = [f for f in files if "tests/api" not in str(f)]
groups = mod.balance(files, mod.CHUNKS)
for i, g in enumerate(groups[:2], 1):
    print(f"chunk-{i}: {len(g)} files, {sum(f.stat().st_size for f in g)} bytes")
print("CHUNK1_FILES")
for f in groups[0]:
    print(f)

#!/usr/bin/env python
"""Round-7 audit (100x-122): every public symbol of every pharmaml module,
grepped against playground.py. Zero-hit symbols are candidates for gaps.

Must run with cwd = pharma-ml-toolkit.
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys

MODULES = [
    "pharmaml.chem",
    "pharmaml.ml",
    "pharmaml.docking",
    "pharmaml.pharma",
    "pharmaml.biology",
    "pharmaml.generative",
    "pharmaml.viz",
]

PLAYGROUND = open("playground.py", encoding="utf-8").read()

total = 0
zero = []
for mod_name in MODULES:
    mod = importlib.import_module(mod_name)
    names = [n for n in dir(mod) if not n.startswith("_")]
    symbols = []
    for n in names:
        obj = getattr(mod, n)
        if inspect.ismodule(obj):
            continue
        symbols.append(n)
    hits = 0
    misses = []
    for n in symbols:
        total += 1
        if re.search(rf"\b{re.escape(n)}\b", PLAYGROUND):
            hits += 1
        else:
            misses.append(n)
    print(f"{mod_name}: {len(symbols)} public symbols, {hits} hit, {len(misses)} zero-hit")
    for m in misses:
        zero.append((mod_name, m))

print(f"\nTOTAL {total} symbols; zero-hit {len(zero)}:")
for mod_name, m in zero:
    print(f"  {mod_name}.{m}")
sys.exit(0)

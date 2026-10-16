"""Probe 5: brute-force locate the stderr leak (RDKit noise escaping the
intended silencing) per (parent, seed, draw) with C-level fd capture."""
from __future__ import annotations

import os
import random
import tempfile

from pharmaml.generative.corpus import DEFAULT_CORPUS
from pharmaml.generative.genetic import _build_decoration_pool, _scaffold_mutate

pool = _build_decoration_pool(list(DEFAULT_CORPUS))
parents = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CN1CCCC1c1cccnc1", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "c1ccc([C@H](C)N)cc1", "C[N+](C)(C)c1ccccc1", "c1ccc(S(=O)(=O)N)cc1",
    "c1ccc(B(O)O)cc1", "Cn1cccc1", "OC(=O)[C@@H]1CCCN1",
    "Cc1c[nH]c(=O)[nH]c1=O", "c1ccc2c(c1)ncnc2N",
]

leaks = []
tmp = tempfile.NamedTemporaryFile(delete=False)
for parent in parents:
    for seed in range(8):
        rng = random.Random(seed)
        for draw in range(60):
            # capture C-level stderr for exactly this call
            with open(tmp.name, "wb") as sink:
                saved = os.dup(2)
                os.dup2(sink.fileno(), 2)
                try:
                    out = _scaffold_mutate(parent, pool, rng, 120)
                finally:
                    os.dup2(saved, 2)
                    os.close(saved)
            with open(tmp.name, "rb") as src:
                content = src.read().decode(errors="replace")
            if content.strip():
                leaks.append((parent, seed, draw, out, content[:200]))

print(f"leaking calls: {len(leaks)}")
for parent, seed, draw, out, content in leaks[:10]:
    print(f"  parent={parent} seed={seed} draw={draw} out={out!r}")
    print(f"  stderr={content!r}")

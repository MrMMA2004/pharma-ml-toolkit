"""Probe 4: stderr noise leak in scaffold mode (nested RDLogger toggles).

`scaffold_mutate` disables rdApp.error at entry, but the nested
`_scaffold_split` call RE-ENABLES it in its `finally` on every cache miss
(seeded runs: every seed molecule is a miss).  Later sanitisation failures
inside `_scaffold_mutate` then leak raw RDKit noise to stderr, defeating the
"keep the loop quiet" intent of the code's own comment.
Run with a FRESH interpreter: `python probe_gen_v2_4_stderr.py 2>err.txt`.
"""
from __future__ import annotations

import random
import sys

from pharmaml.generative.genetic import _build_decoration_pool, _scaffold_mutate

# parent chosen to produce at least one kekulize-failing mutant
PARENT = "c1ccc2c(c1)ncnc2N"
pool = _build_decoration_pool(["CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1"])
rng = random.Random(0)
outs = []
for _ in range(200):
    m = _scaffold_mutate(PARENT, pool, rng, 120)
    outs.append(m)
n_none = sum(1 for m in outs if m is None)
print(f"mutants: {sum(1 for m in outs if m is not None)}, None: {n_none}")
sys.stderr.flush()
print("NOTE: any '[ERROR] Can't kekulize' lines above/below this line on stderr "
      "are the leak (they should have been silenced).")

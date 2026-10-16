"""Probe 3: molzip bond-order conflicts — carbonyl ring systems.

Pool entries derived from ring carbonyls (caffeine lactam =O) carry a
DOUBLE-bonded dummy ``*=O``; normal decorations carry single-bonded dummies.
What happens when each kind is attached to a core whose dummy bond order is
the OTHER kind?  Silent wrong bond orders would be a real chemical-correctness
defect; sanitize-failure -> None -> fallback is acceptable robustness.
"""
from __future__ import annotations

import random

from rdkit import Chem, RDLogger

from pharmaml.generative.corpus import DEFAULT_CORPUS
from pharmaml.generative.genetic import _build_decoration_pool, _scaffold_mutate, _scaffold_split

RDLogger.DisableLog("rdApp.error")

pool = _build_decoration_pool(list(DEFAULT_CORPUS))
print(f"pool size {len(pool)}")
dbl = [p for p in pool if "=" in p.split("*")[0] or p.startswith("*=") or "=*" in p]
print(f"entries with double-bonded dummy: {dbl}")
# how does a carbonyl-O decoration actually look?
for p in pool:
    if p.endswith("=O") or "=O" in p:
        pass
print("pool entries containing '=O':", [p for p in pool if "=O" in p][:10])

# split some ring-carbonyl parents
for parent in ("O=C1CCCCC1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "O=C1c2ccccc2CC1",
               "O=c1[nH]cccc1", "O=C1CC=CCC1"):
    split = _scaffold_split(parent)
    print(f"\nparent {parent!r} -> {split}")

# direct molzip conflict experiments
print("\n--- direct molzip bond-order experiments ---")


def try_zip(core_smi: str, frag_smi: str) -> None:
    try:
        core = Chem.MolFromSmiles(core_smi)
        side = Chem.MolFromSmiles(frag_smi)
        if core is None or side is None:
            print(f"  {core_smi} + {frag_smi}: PARSE FAIL")
            return
        zipped = Chem.molzip(core, side)
        Chem.SanitizeMol(zipped)
        print(f"  {core_smi} + {frag_smi} -> {Chem.MolToSmiles(zipped)}")
    except Exception as exc:
        print(f"  {core_smi} + {frag_smi} -> EXC {type(exc).__name__}: {str(exc)[:60]}")


# single-bond core dummy vs double-bond fragment dummy (and vice versa)
try_zip("c1ccc([*:1])cc1", "[*:1]=O")     # aromatic C + carbonyl-O decoration
try_zip("C1CCCCC1([*:1])", "[*:1]=O")     # sp3 ring C + carbonyl-O decoration
try_zip("O=[*:1]C1CCCCC1", "[*:1]O")      # carbonyl core + hydroxyl decoration
try_zip("O=[*:1]C1CCCCC1", "[*:1]C")      # carbonyl core + methyl decoration
try_zip("c1ccc([*:1])cc1", "[*:1]C")      # control: aromatic + methyl
try_zip("C1CCCCC1([*:1])", "[*:1]O")      # control: sp3 + hydroxyl
try_zip("c1ccc([*:1])cc1", "[*:1]=S")     # aromatic C + thiocarbonyl

# --- now the GA-level view: run _scaffold_mutate on ring-carbonyl parents ---
print("\n--- _scaffold_mutate on ring-carbonyl parents (default pool) ---")
rng = random.Random(11)
for parent in ("O=C1CCCCC1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Nc1ccc(O)cc1",
               "O=C1c2ccccc2CC1"):
    outs: dict[str, int] = {}
    nones = 0
    for _ in range(200):
        m = _scaffold_mutate(parent, pool, rng, 120)
        if m is None:
            nones += 1
        else:
            outs[m] = outs.get(m, 0) + 1
    print(f"parent {parent!r}: {len(outs)} distinct, {nones} None (of 200)")
    for m, c in sorted(outs.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    x{c:3d} {m}")

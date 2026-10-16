"""Audit pharmaml's element weight table against RDKit's periodic table.

Round 100x-22-c work item: the 19-a/20-a triage list flagged Cu/Ni entries.
This probe compares EVERY entry of ``_ATOMIC_WEIGHTS`` against
``Chem.GetPeriodicTable().GetAtomicWeight`` and lists elements that are
missing from our table but present in RDKit (restricted to elements with
Z <= 53 for readability, plus the pharma-relevant heavier ones).
"""

from rdkit import Chem

from pharmaml.pharma.units import _ATOMIC_WEIGHTS

pt = Chem.GetPeriodicTable()

print(f"{'el':>4} {'ours':>10} {'rdkit':>12} {'absdiff':>10}  verdict")
mismatches = []
for symbol, ours in _ATOMIC_WEIGHTS.items():
    z = pt.GetAtomicNumber(symbol)
    rdkit_w = pt.GetAtomicWeight(z)
    diff = abs(ours - rdkit_w)
    verdict = "OK" if diff <= 0.01 else "MISMATCH"
    if verdict == "MISMATCH":
        mismatches.append((symbol, ours, rdkit_w, diff))
    print(f"{symbol:>4} {ours:>10.4f} {rdkit_w:>12.6f} {diff:>10.6f}  {verdict}")

print(f"\nentries audited: {len(_ATOMIC_WEIGHTS)}, mismatches: {len(mismatches)}")
for row in mismatches:
    print("  MISMATCH:", row)

# Which elements are MISSING from our table (pharma-relevant window)?
print("\nmissing elements (Z = 1..53) vs our table:")
missing = []
for z in range(1, 54):
    symbol = pt.GetElementSymbol(z)
    if symbol not in _ATOMIC_WEIGHTS:
        missing.append((z, symbol, pt.GetAtomicWeight(z)))
for z, symbol, w in missing:
    print(f"  Z={z:>3} {symbol:>3} rdkit weight {w:.4f}")
print(f"total missing in Z<=53 window: {len(missing)}")

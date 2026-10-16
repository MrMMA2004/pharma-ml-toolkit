"""Probe 1: does _scaffold_split round-trip EXACTLY across DEFAULT_CORPUS?

Docstring claim (genetic.py): "so Chem.molzip(core, decorations) rebuilds
exactly the parent molecule".  Test every corpus entry plus a battery of
chemically nasty parents (stereo, charges, isotopes, bridging decorations,
multi-dummy fragments, fused/spiro systems, explicit hydrogens).
"""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmaml.generative.corpus import DEFAULT_CORPUS
from pharmaml.generative.genetic import _scaffold_split

RDLogger.DisableLog("rdApp.error")


def reassemble(core_smi: str, frag_smis: list[str]) -> str | None:
    try:
        core = Chem.MolFromSmiles(core_smi)
        side = Chem.MolFromSmiles(".".join(frag_smis))
        if core is None or side is None:
            return "PARSE_FAIL"
        zipped = Chem.molzip(core, side)
        Chem.SanitizeMol(zipped)
        return Chem.MolToSmiles(zipped)
    except Exception:
        return "EXC"


def check(parent: str) -> tuple[str, str, str]:
    split = _scaffold_split(parent)
    if split is None:
        return ("nosplit", "", "")
    core_smi, frag_smis = split
    rebuilt = reassemble(core_smi, list(frag_smis))
    canonical_parent = Chem.MolToSmiles(Chem.MolFromSmiles(parent))
    return ("OK" if rebuilt == canonical_parent else "MISMATCH", rebuilt, canonical_parent)


# --- Part A: the whole DEFAULT_CORPUS -------------------------------------
failures = []
skipped = 0
for smi in DEFAULT_CORPUS:
    status, rebuilt, parent = check(smi)
    if status == "nosplit":
        skipped += 1
    elif status != "OK":
        failures.append((smi, status, rebuilt, parent))

print(f"corpus entries: {len(DEFAULT_CORPUS)}; no-split (acyclic/bare/None): {skipped}")
print(f"round-trip failures: {len(failures)}")
for smi, status, rebuilt, parent in failures[:40]:
    print(f"  {status}: {smi}\n    rebuilt={rebuilt}\n    parent ={parent}")

# how many were actually split (i.e. the claim's coverage)
n_split = sum(1 for s in DEFAULT_CORPUS if _scaffold_split(s) is not None)
print(f"actually split: {n_split}")

# --- Part B: nasty parents --------------------------------------------------
nasty = {
    # chiral decoration attached to a scaffold (stereocenter in side chain)
    "chiral_sidechain": "c1ccc([C@H](C)N)cc1",
    "chiral_sidechain2": "c1ccc([C@@H](N)(C))cc1",
    # chiral scaffold atoms + decorations (ibuprofen-like)
    "ibuprofen_like": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    # two decorations on one atom
    "geminal_two_deco": "c1ccc(C(F)(Cl)Br)cc1",
    # bridging decoration: -O-CH2-O- across two aromatic positions
    "methylenedioxy": "COc1ccc2c(c1)OCOc1ccccc21",
    "benzodioxole_plain": "OCC1OCc2ccccc21",
    # charged decorations
    "quaternary_N": "C[N+](C)(C)c1ccccc1",
    "zwitterion": "C[N+](C)(C)CC(=O)[O-]",
    "carboxylate": "c1ccc(C(=O)[O-])cc1",
    # isotope decorations
    "isotope": "c1ccc([13CH3])cc1",
    "isotope2": "[2H]c1ccccc1",
    # N-substituted heteroaromatic (drop -> [nH])
    "n_methyl_indole": "Cn1cccc1",
    "n_methyl_imidazole": "Cn1ccnc1",
    # amide N decorations on ring N
    "n_acetyl_piperidine": "CC(=O)N1CCCCC1",
    # sulfonyl / nitro decorations
    "sulfonamide": "Cc1ccc(S(=O)(=O)N)cc1",
    "nitro": "Cc1ccc([N+](=O)[O-])cc1",
    # phosphorus / silicon decorations
    "phosphate": "c1ccc(OP(=O)(O)O)cc1",
    "silane": "c1ccc([Si](C)(C)C)cc1",
    # boron
    "boronic_acid": "c1ccc(B(O)O)cc1",
    # stereo in ring + decoration
    "stereo_ring": "C[C@H]1CCCC[C@H]1c1ccccc1",
    # stereo double bond in decoration
    "e_double_bond": "c1ccc(/C=C/C)cc1",
    "z_double_bond": "c1ccc(/C=C\\C)cc1",
    # tautomer-ish
    "keto_enol": "CC(=O)c1ccccc1O",
    # explicit H atom in decoration
    "explicit_h": "c1ccc([H])cc1",
    # multi-fragment-bridge: fused bicyclic with epoxide on ring
    "epoxide_on_ring": "C1COC1c1ccccc1",
    "benzene_oxide": "c1ccccc1O1",
    # long decoration chains
    "long_deco": "c1ccc(CCCCCCCC(=O)O)cc1",
    # multiple identical decorations (symmetric)
    "symmetric": "Cc1ccc(C)cc1C",
    "tri_methyl": "Cc1c(C)c(C)c(C)c(C)c1C",
    # amino acids (zwitterionic & chiral)
    "phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "proline": "OC(=O)[C@@H]1CCCN1",
    # nucleobases
    "adenine": "Nc1ncnc2c1ncn2",
    "thymine": "Cc1c[nH]c(=O)[nH]c1=O",
    # drugs
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "nicotine": "CN1CCCC1c1cccnc1",
}
print("\n--- nasty parents ---")
for label, smi in nasty.items():
    status, rebuilt, parent = check(smi)
    mark = "ok " if status == "OK" else ("-- " if status == "nosplit" else "FAIL")
    print(f"[{mark}] {label:22s} {smi}")
    if status == "MISMATCH":
        print(f"        rebuilt={rebuilt}")
        print(f"        parent ={parent}")

# --- Part C: label determinism across repeated calls / cache ---------------
s = _scaffold_split("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O")
_scaffold_split.cache_clear()
s2 = _scaffold_split("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O")
print(f"\nlabel determinism across cache clears: {s == s2}")

# --- Part D: do decorations with >1 dummy occur at all in the corpus? -------
multi = []
for smi in DEFAULT_CORPUS:
    sp = _scaffold_split(smi)
    if sp is None:
        continue
    for frag in sp[1]:
        if frag.count("[*") > 1:
            multi.append((smi, frag))
print(f"corpus entries with multi-dummy (bridging) decorations: {len(multi)}")
for smi, frag in multi[:10]:
    print(f"  {smi}  ->  {frag}")

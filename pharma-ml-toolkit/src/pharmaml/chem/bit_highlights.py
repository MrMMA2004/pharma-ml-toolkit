r"""Multi-bit ECFP highlight rendering and bit-presence screening.

Round 174 absorbed ``precisionmatics/ECFP6-Highlighter-GUI`` (the
single-file PyQt5 app).  The GUI shell is the not-vendored part; the
two ideas transfer:

* **The union bit highlight** — ``ECFP6_Gui.py``'s
  ``find_highlight_atoms_bonds`` (lines 428-454): for every requested
  bit, expand each ``(center_atom, radius)`` hit with
  ``Chem.FindAtomEnvironmentOfRadiusN`` and accumulate the atoms and
  bonds into one set, then draw the parent molecule with
  ``rdMolDraw2D.PrepareAndDrawMolecule``.  This module upgrades the
  idea with **per-bit palettes** (the GUI's dead ``matplotlib.colors``
  import betrays that unfinished intent) and uses the modern
  ``setHighlightColour`` API (the GUI's ``options.highlightColor``
  spelling is broken on current RDKit).
* **The bit presence report** — the set intersection against the
  molecule's on-bits (``present``/``missing`` per molecule), which
  screens a library against the bits a model cares about.

The fingerprinting itself reuses the modern
``rdFingerprintGenerator.GetMorganGenerator`` + ``AdditionalOutput``
route (the GUI's deprecated ``GetMorganFingerprintAsBitVect`` call and
its pre-fingerprint kekulization -- which verifiably changes the
on-bit set -- are deliberately NOT copied).

Examples
--------
>>> from pharmaml.chem.bit_highlights import bit_environments, bit_presence_report
>>> smiles = "CC(=O)Oc1ccccc1C(=O)O"           # aspirin
>>> atoms, bonds = bit_environments(smiles, bits=[389, 456, 541])
>>> len(atoms) >= 4 and len(bonds) >= 3        # real environments
True
>>> report = bit_presence_report([smiles, "CCO"], bits=[389, 456, 541])
>>> list(report["n_present"])
[3, 0]
>>> bool(report["missing"].iloc[1] == [389, 456, 541])
True
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "bit_environments",
    "bit_presence_report",
    "highlight_bits_png",
]

# the palette the GUI wanted (matplotlib's tab10 first arc) -- one colour
# per requested bit so overlapping environments stay readable
_BIT_PALETTE = (
    (0.35, 0.70, 0.90),   # blue
    (0.90, 0.60, 0.30),   # orange
    (0.40, 0.75, 0.40),   # green
    (0.85, 0.35, 0.35),   # red
    (0.70, 0.55, 0.85),   # purple
    (0.75, 0.75, 0.35),   # olive
    (0.35, 0.80, 0.80),   # cyan
    (0.90, 0.55, 0.75),   # pink
)


def _morgan_with_bitinfo(mol: object, radius: int, n_bits: int) -> tuple[object, dict]:
    """Fingerprint + ``{bit: ((center, radius), ...)}`` via the generator API."""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    fingerprint = generator.GetFingerprint(mol, additionalOutput=output)
    bit_map = output.GetBitInfoMap()  # {bit: ((atom, radius), ...)}
    return fingerprint, bit_map


def bit_environments(
    smiles: str,
    bits: Sequence[int],
    radius: int = 3,
    n_bits: int = 2048,
) -> tuple[set[int], set[int]]:
    """The union of atom/bond environments of the requested bits.

    Every ``(center_atom, radius)`` hit of every present bit expands
    with ``Chem.FindAtomEnvironmentOfRadiusN``; atoms and bonds
    accumulate across bits (the highlighter's union semantics;
    radius-0 hits highlight the center atom alone).

    Examples
    --------
    >>> atoms, bonds = bit_environments("CC(=O)Oc1ccccc1C(=O)O", bits=[389, 456, 541])
    >>> sorted(atoms)[:3]
    [1, 3, 4]
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise DataValidationError(f"bit_environments: unparseable SMILES {smiles!r}")
    _fingerprint, bit_map = _morgan_with_bitinfo(mol, radius, n_bits)
    atoms: set[int] = set()
    bonds: set[int] = set()
    for bit in bits:
        for hit in bit_map.get(int(bit), ()):  # (center, radius) tuples
            center, hit_radius = int(hit[0]), int(hit[1])
            env_bonds = chem.FindAtomEnvironmentOfRadiusN(mol, hit_radius, center)
            for bond_idx in env_bonds:
                bond = mol.GetBondWithIdx(int(bond_idx))
                bonds.add(int(bond_idx))
                atoms.add(int(bond.GetBeginAtomIdx()))
                atoms.add(int(bond.GetEndAtomIdx()))
            atoms.add(center)
    return atoms, bonds


def highlight_bits_png(
    smiles: str,
    bits: Sequence[int],
    radius: int = 3,
    n_bits: int = 2048,
    size: tuple[int, int] = (420, 380),
    atom_radii: float = 0.4,
) -> bytes:
    """Render the molecule with the requested bits highlighted (PNG bytes).

    Per-bit palette colours (the upgrade over the GUI's single red),
    ``PrepareAndDrawMolecule`` with atom/bond colour dictionaries, and
    the modern ``setHighlightColour``-safe API.  Bits absent from the
    molecule are skipped (the GUI's present/missing set arithmetic).

    Examples
    --------
    >>> data = highlight_bits_png("CC(=O)Oc1ccccc1C(=O)O", bits=[389, 456, 541])
    >>> data[:8] == b"\\x89PNG\\r\\n\\x1a\\n"       # the PNG magic
    True
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise DataValidationError(f"highlight_bits_png: unparseable SMILES {smiles!r}")
    _fingerprint, bit_map = _morgan_with_bitinfo(mol, radius, n_bits)
    atom_colors: dict[int, tuple[float, float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float, float]] = {}
    for position, bit in enumerate(bits):
        color = _BIT_PALETTE[position % len(_BIT_PALETTE)] + (0.55,)  # RGBA overlay
        for hit in bit_map.get(int(bit), ()):
            center, hit_radius = int(hit[0]), int(hit[1])
            env_bonds = chem.FindAtomEnvironmentOfRadiusN(mol, hit_radius, center)
            for bond_idx in env_bonds:
                idx = int(bond_idx)
                bond_colors[idx] = color
                bond = mol.GetBondWithIdx(idx)
                atom_colors[int(bond.GetBeginAtomIdx())] = color
                atom_colors[int(bond.GetEndAtomIdx())] = color
            atom_colors[center] = color
    AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    _ = options  # the modern API colours via the dicts below
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atom_colors),
        highlightBonds=list(bond_colors),
        highlightAtomColors={idx: color[:3] for idx, color in atom_colors.items()},
        highlightBondColors={idx: color[:3] for idx, color in bond_colors.items()},
        highlightAtomRadii={idx: atom_radii for idx in atom_colors},
    )
    drawer.FinishDrawing()
    return bytes(drawer.GetDrawingText())


def bit_presence_report(
    smiles_list: Sequence[str],
    bits: Sequence[int],
    radius: int = 3,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Screen a molecule list against a bit set (the GUI's report table).

    Per molecule: the present bits (intersection with the on-bits),
    the missing bits, the counts and the coverage fraction -- "which
    library compounds carry the model's top bits".

    Examples
    --------
    >>> report = bit_presence_report(["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
    ...                              bits=[389, 456, 541])
    >>> list(report["n_present"])
    [3, 0]
    >>> float(report["coverage"].iloc[0])
    1.0
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    wanted = [int(b) for b in bits]
    if not wanted:
        raise DataValidationError("bit_presence_report: bits must be non-empty")
    rows = []
    for smiles in smiles_list:
        mol = chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({"smiles": smiles, "present": [], "missing": wanted,
                         "n_present": 0, "n_missing": len(wanted), "coverage": 0.0,
                         "valid": False})
            continue
        fingerprint, _bit_map = _morgan_with_bitinfo(mol, radius, n_bits)
        on_bits = set(fingerprint.GetOnBits())  # type: ignore[attr-defined]
        present = [b for b in wanted if b in on_bits]
        missing = [b for b in wanted if b not in on_bits]
        rows.append({
            "smiles": smiles,
            "present": present,
            "missing": missing,
            "n_present": len(present),
            "n_missing": len(missing),
            "coverage": len(present) / len(wanted),
            "valid": True,
        })
    return pd.DataFrame(rows)

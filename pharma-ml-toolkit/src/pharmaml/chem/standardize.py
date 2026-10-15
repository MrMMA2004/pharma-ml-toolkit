"""Molecular standardization and 3D ligand generation.

Upgrades ``others/Docking/ligand generation/nigga.py`` and ``nigga2.py``
into one typed, single-ligand entry point.  The legacy pipeline was:

SMILES parse -> tautomer canonicalization
(:class:`rdkit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator`)
-> skip molecules with unsupported metals (Fe, Zn, Cu, Mn, Co, Ni, Mo, W)
-> ``AddHs`` -> ETKDG multi-conformer embedding -> MMFF optimisation
(UFF fallback) -> keep the lowest-energy conformer -> write a PDB file
with CONECT connectivity records.

Differences to the rar originals:

* **No MOL2 leg.**  The scripts wrote ``.mol2`` through OpenBabel with
  Gasteiger charges; OpenBabel is not a pharmaml dependency, so that
  step is intentionally dropped.  For docking-ready 3D formats use
  :func:`pharmaml.docking.smiles_to_pdbqt` (RDKit+Meeko PDBQT) or
  :func:`pharmaml.chem.table_to_sdf` (embedded multi-conformer SDF
  libraries) instead.
* **No multiprocessing.**  The ``Pool`` fan-out over a CSV of ligands
  is replaced by one deterministic function per ligand; batch PDBQT
  campaigns are served by
  :func:`pharmaml.docking.prepare_ligand_library`.
* **Typed errors.**  Parse failures, unsupported metals and embedding
  failures raise :class:`~pharmaml.exceptions.DataValidationError`
  instead of being logged to ``unsupported_molecules.log`` and skipped.
* **Determinism.**  The ETKDG random seed is pinned (42), matching the
  ``random_state=42`` convention used across the ML registry, so the
  same SMILES always yields the same conformers.

RDKit remains an optional dependency: both public functions import it
lazily and raise an actionable
:class:`~pharmaml.exceptions.DependencyError` when it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory

_LOG = get_logger(__name__)

__all__ = [
    "UNSUPPORTED_METALS",
    "PdbLigandReport",
    "require_rdkit_standardize",
    "canonicalize_tautomer",
    "smiles_to_pdb",
]

#: Metal atoms the rar scripts refused to process because MMFF/UFF
#: parameterisation is unreliable for them.  Molecules containing any
#: of these are rejected by :func:`smiles_to_pdb` with a message naming
#: the offending element.
UNSUPPORTED_METALS: frozenset[str] = frozenset(
    {"Fe", "Zn", "Cu", "Mn", "Co", "Ni", "Mo", "W"}
)

#: Pinned ETKDG random seed (reproducible conformers, package convention).
_RANDOM_SEED = 42


def require_rdkit_standardize():
    """Import and return the RDKit ``Chem`` module or raise helpfully."""
    try:
        from rdkit import Chem  # noqa: PLC0415 -- deliberate lazy import
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for molecular standardization and 3D ligand "
            "generation. Install it with:  pip install 'pharmaml[chem]'  "
            "(or pip install rdkit)"
        ) from exc
    return Chem


def canonicalize_tautomer(smiles: str) -> str:
    """Return the canonical-tautomer SMILES of ``smiles``.

    Runs the RDKit ``TautomerEnumerator`` used by the original ligand
    generation scripts, so keto/enol pairs such as
    ``"CC(=O)CC(C)=O"`` and ``"CC(O)=CC(C)=O"`` map to the same
    canonical string.

    Parameters
    ----------
    smiles:
        Input SMILES.

    Returns
    -------
    str
        SMILES of the canonical tautomer.

    Raises
    ------
    DataValidationError
        If the SMILES cannot be parsed or the tautomer enumeration
        fails.
    DependencyError
        If RDKit is not installed.
    """
    Chem = require_rdkit_standardize()
    from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: PLC0415

    # Blank/non-str input must raise "Cannot parse SMILES" (see smiles_to_pdb).
    mol = (
        Chem.MolFromSmiles(smiles)
        if isinstance(smiles, str) and smiles.strip()
        else None
    )
    if mol is None:
        raise DataValidationError(f"Cannot parse SMILES: {smiles!r}")
    try:
        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception as exc:  # RDKit raises assorted Boost/ValueError types
        raise DataValidationError(
            f"Tautomer canonicalization failed for {smiles!r}: {exc}"
        ) from exc
    return Chem.MolToSmiles(canonical)


@dataclass(frozen=True, slots=True)
class PdbLigandReport:
    """Audit record for one :func:`smiles_to_pdb` conversion.

    Attributes
    ----------
    name:
        Molecule name recorded in the PDB ``COMPND`` record.
    smiles:
        Input SMILES (as given, pre-canonicalization).
    output_path:
        Resolved path of the written PDB file.
    num_conformers:
        Number of conformers that were embedded and optimised.
    forcefield:
        Force field used, ``"MMFF"`` or ``"UFF"`` (the fallback).
    num_atoms:
        Atom count of the exported conformer (explicit hydrogens
        included).
    num_bonds:
        Bond count of the exported conformer.
    """

    name: str
    smiles: str
    output_path: str
    num_conformers: int
    forcefield: str
    num_atoms: int
    num_bonds: int

    def status(self) -> str:
        """One-line summary, mirroring the per-ligand log lines of the
        original scripts (``[name] ... optimized with ... saved ...``)."""
        return (
            f"[{self.name}] {self.num_conformers} conformer(s) embedded, "
            f"optimized with {self.forcefield}, {self.num_atoms} atoms / "
            f"{self.num_bonds} bonds -> {self.output_path}"
        )


def _apply_default_b_factor(pdb_block: str, default_b_factor: float) -> str:
    """Overwrite the B-factor column (61-66) of ATOM/HETATM records.

    RDKit's PDB writer always emits ``0.00``; the rar scripts instead
    wrote ``default_b_factor=20.0`` (the AutoDock convention), so the
    field is rewritten with fixed-column surgery that keeps every other
    byte of the record untouched.
    """
    lines = pdb_block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            lines[i] = f"{line[:60]}{default_b_factor:6.2f}{line[66:]}"
    return "\n".join(lines) + "\n"


def smiles_to_pdb(
    smiles: str,
    output_path: str | Path,
    name: str = "LIGAND",
    num_confs: int = 50,
    canonicalize: bool = True,
    default_b_factor: float = 20.0,
) -> PdbLigandReport:
    """Generate an optimised 3D PDB ligand from a SMILES string.

    Full ``nigga.py`` pipeline, single-ligand edition: parse, optional
    tautomer canonicalization, unsupported-metal screening, explicit
    hydrogens, ETKDG embedding of ``num_confs`` conformers, MMFF (or
    UFF fallback) optimisation of every conformer, and export of the
    lowest-energy conformer as a PDB file with CONECT connectivity
    records (written by RDKit's PDB writer).

    Parameters
    ----------
    smiles:
        Input SMILES.
    output_path:
        Destination ``.pdb`` file (parent directories are created).
    name:
        Molecule name stored in the ``_Name`` property / ``COMPND``
        record.
    num_confs:
        Number of ETKDG conformers to embed (default 50, as in the
        original scripts).
    canonicalize:
        Run tautomer canonicalization before embedding (default
        ``True``); pass ``False`` to keep the input tautomer as-is.
    default_b_factor:
        B-factor written into every ATOM/HETATM record (default 20.0,
        the AutoDock convention of the rar scripts).

    Returns
    -------
    PdbLigandReport
        Audit record; ``forcefield`` reports ``"MMFF"`` or ``"UFF"``.

    Raises
    ------
    DataValidationError
        If the SMILES cannot be parsed, the molecule contains an
        unsupported metal (the message names the element), embedding
        yields no conformers, or optimisation fails for all conformers.
    DependencyError
        If RDKit is not installed.
    """
    Chem = require_rdkit_standardize()
    from rdkit.Chem import AllChem  # noqa: PLC0415
    from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: PLC0415

    if num_confs < 1:
        raise DataValidationError(
            f"num_confs must be at least 1, got {num_confs}"
        )

    # RDKit parses "" as a *valid* 0-atom molecule (the embedder then dies
    # with "molecule has no atoms") and MolFromSmiles(None) raises a raw
    # Boost error; both must surface as the documented parse failure.
    # Same guard family as chem/descriptors._smiles_to_molecules.
    mol = (
        Chem.MolFromSmiles(smiles)
        if isinstance(smiles, str) and smiles.strip()
        else None
    )
    if mol is None:
        raise DataValidationError(f"Cannot parse SMILES: {smiles!r}")
    mol.SetProp("_Name", name)

    if canonicalize:
        try:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        except Exception as exc:
            raise DataValidationError(
                f"Tautomer canonicalization failed for {smiles!r}: {exc}"
            ) from exc
        mol.SetProp("_Name", name)

    metals = sorted(
        {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in UNSUPPORTED_METALS}
    )
    if metals:
        raise DataValidationError(
            f"Molecule {name!r} contains unsupported metal atom(s) {metals}; "
            f"MMFF/UFF parameterisation is unreliable for {sorted(UNSUPPORTED_METALS)}"
        )

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDG()
    params.randomSeed = _RANDOM_SEED
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        raise DataValidationError(
            f"ETKDG embedding returned no conformers for {name!r} ({smiles!r})"
        )

    energies: list[tuple[int, float]] = []
    forcefield = ""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in conf_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                ff.Minimize()
                energies.append((conf_id, ff.CalcEnergy()))
            except Exception:  # noqa: BLE001 -- skip failing conformers, like the original
                continue
        if energies:
            forcefield = "MMFF"
    if not energies:  # UFF fallback (also the only route when MMFF lacks params)
        for conf_id in conf_ids:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                energies.append((conf_id, ff.CalcEnergy()))
            except Exception:  # noqa: BLE001
                continue
        if energies:
            forcefield = "UFF"
    if not energies:
        raise DataValidationError(
            f"Optimization failed for all {len(conf_ids)} conformers of {name!r}"
        )

    lowest_id = min(energies, key=lambda pair: pair[1])[0]
    best = Chem.Mol(mol)
    best.RemoveAllConformers()
    best.AddConformer(mol.GetConformer(lowest_id), assignId=True)
    best.SetProp("_Name", name)

    num_atoms = best.GetNumAtoms()
    num_bonds = best.GetNumBonds()
    pdb_block = _apply_default_b_factor(Chem.MolToPDBBlock(best), default_b_factor)

    file_path = Path(output_path)
    ensure_directory(file_path.parent)
    file_path.write_text(pdb_block, encoding="utf-8")

    report = PdbLigandReport(
        name=name,
        smiles=smiles,
        output_path=str(file_path),
        num_conformers=len(conf_ids),
        forcefield=forcefield,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
    )
    _LOG.info("%s", report.status())
    return report

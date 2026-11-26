"""Molecular standardization and 3D ligand generation.

Upgrades ``others/Docking/ligand generation/ligprep.py`` and ``ligprep2.py``
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

Examples
--------
>>> from pharmaml.chem.standardize import (
...     UNSUPPORTED_METALS, canonicalize_tautomer)
>>> canonicalize_tautomer("CC(=O)CC(C)=O")   # the keto form ...
'CC(=O)CC(C)=O'
>>> canonicalize_tautomer("CC(O)=CC(C)=O")   # ... and its enol agree
'CC(=O)CC(C)=O'
>>> sorted(UNSUPPORTED_METALS)               # screened out by smiles_to_pdb
['Co', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'W', 'Zn']
>>> canonicalize_tautomer("not-a-smiles")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Cannot parse SMILES: 'not-a-smiles'

>>> from pathlib import Path
>>> from pharmaml.chem.standardize import smiles_to_pdb
>>> pdb_path = Path("/tmp/pharmaml_doctest_standardize.pdb")
>>> report = smiles_to_pdb("CCO", pdb_path, name="ETHANOL", num_confs=3)
>>> (report.num_conformers, report.forcefield, report.num_atoms, report.num_bonds)
(3, 'MMFF', 9, 8)
>>> lines = pdb_path.read_text().splitlines()
>>> lines[0]                                 # COMPND records the caller's name
'COMPND    ETHANOL'
>>> sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))  # explicit H
9
>>> {line[60:66] for line in lines if line.startswith(("ATOM", "HETATM"))}
{' 20.00'}
>>> sum(1 for line in lines if line.startswith("CONECT"))   # connectivity
3
>>> pdb_path.unlink()

>>> smiles_to_pdb("CCO", "/tmp/pharmaml_doctest_standardize.pdb", num_confs=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: num_confs must be at least 1, got 0
>>> smiles_to_pdb("C1CC", "/tmp/pharmaml_doctest_standardize.pdb")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Cannot parse SMILES: 'C1CC'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory

_LOG = get_logger(__name__)

__all__ = [
    "NEUTRALIZE_PATTERNS",
    "REINVENT_DEFAULT_ELEMENTS",
    "UNSUPPORTED_METALS",
    "PdbLigandReport",
    "canonicalize_tautomer",
    "reinvent_standardize_mol",
    "reinvent_standardize_smiles",
    "require_rdkit_standardize",
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
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for molecular standardization and 3D ligand "
            "generation. From the toolkit package root, run:  python -m pip install -e '.[chem]'"
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
    from rdkit.Chem.MolStandardize import rdMolStandardize

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

    Full ``ligprep.py`` pipeline, single-ligand edition: parse, optional
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
    # AllChem only re-exports these; importing from the defining modules
    # gives the fully-typed names in rdkit-stubs (identical objects).
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    from rdkit.Chem.MolStandardize import rdMolStandardize

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

    params = rdDistGeom.ETKDG()
    # rdkit-stubs declares EmbedParameters.randomSeed with a broken
    # property signature (untyped getter/setter pair), so mypy rejects
    # the assignment; setting the seed on the params object is the
    # documented runtime API.
    params.randomSeed = _RANDOM_SEED  # type: ignore[assignment]
    conf_ids = list(
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    )
    if not conf_ids:
        raise DataValidationError(
            f"ETKDG embedding returned no conformers for {name!r} ({smiles!r})"
        )

    energies: list[tuple[int, float]] = []
    forcefield = ""
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        for conf_id in conf_ids:
            try:
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                ff.Minimize()
                energies.append((conf_id, ff.CalcEnergy()))
            except Exception:  # noqa: BLE001, S112 — per-conformer FF failures are expected
                continue
        if energies:
            forcefield = "MMFF"
    if not energies:  # UFF fallback (also the only route when MMFF lacks params)
        for conf_id in conf_ids:
            try:
                rdForceFieldHelpers.UFFOptimizeMolecule(mol, confId=conf_id)
                ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf_id)
                energies.append((conf_id, ff.CalcEnergy()))
            except Exception:  # noqa: BLE001, S112 — per-conformer FF failures are expected
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


# ---------------------------------------------------------------------------
# The REINVENT4 standardization chain (round 100x-164)
# ---------------------------------------------------------------------------

#: REINVENT4's neutralisation SMARTS (``reinvent/chemistry/standardization/
#: filter_registry.py::NEUTRALIZE_PATTERNS``) -- every pattern is a charged
#: form replaced by its neutral counterpart (``ReplaceSubstructs``), applied
#: repeatedly until the molecule stops changing:
#: imidazoles, amines, carboxylates/alkoxides, thiolates, sulfonamides,
#: enamines, tetrazolates, sulfoxides and amide anions.
NEUTRALIZE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("[n+;H]", "n"),
    ("[N+;!H0]", "N"),
    ("[$([O-]);!$([O-][#7])]", "O"),
    ("[S-;X1]", "S"),
    ("[$([N-;X2]S(=O)=O)]", "N"),
    ("[$([N-;X2][C,N]=C)]", "N"),
    ("[n-]", "[nH]"),
    ("[$([S-]=O)]", "S"),
    ("[$([N-]C=O)]", "N"),
)

#: The classical Reinvent prior vocabulary's elements (C N O F S Cl Br) --
#: ``valid_size``'s element filter in REINVENT4's default chain.
REINVENT_DEFAULT_ELEMENTS: frozenset[int] = frozenset({6, 7, 8, 9, 16, 17, 35})

#: REINVENT4's unwanted substructures (``_unwanted_patterns``): rejected
#: outright rather than transformed.
_UNWANTED_PATTERNS: tuple[str, ...] = ("[C-]#[N+]", "[OH+]", "[SH]")

#: Cap on the neutralisation loop (REINVENT4 loops until the SMILES stops
#: changing; the cap guards pathological oscillation).
_MAX_NEUTRALIZE_PASSES = 10


def reinvent_standardize_mol(
    mol: Any,
    *,
    min_heavy_atoms: int = 2,
    max_heavy_atoms: int = 70,
    allowed_elements: frozenset[int] | set[int] | None = None,
    neutralize: bool = True,
) -> Any | None:
    """The REINVENT4 ``default`` standardization chain, on an RDKit Mol.

    Ported step by step from ``reinvent/chemistry/standardization/
    filter_registry.py::standardize`` (the chain REINVENT4 applies to
    transfer-learning input and library files):

    1. **largest fragment** (most heavy atoms -- salts become their
       parent ion);
    2. **RemoveHs**;
    3. **salt removal** -- RDKit's built-in ``SaltRemover`` (the default
       salt/solvent SMARTS list, ``dontRemoveEverything=True``);
    4. **neutralisation** -- the nine :data:`NEUTRALIZE_PATTERNS`
       replacements looped to a fixpoint;
    5. **general cleanup** -- ``Chem Cleanup`` + sanitise + RemoveHs;
    6. **unwanted patterns** -- ``[C-]#[N+]``, ``[OH+]``, ``[SH]`` are
       *rejected*, not cleaned;
    7. **valid size** -- heavy atoms strictly between ``min_heavy_atoms``
       and ``max_heavy_atoms`` (REINVENT4's 2..70) and every element in
       ``allowed_elements`` (the prior vocabulary C N O F S S Cl Br).

    Parameters
    ----------
    mol:
        A sanitised (or at least parseable) RDKit ``Mol``.
    min_heavy_atoms, max_heavy_atoms:
        The heavy-atom window (exclusive bounds, REINVENT4's defaults).
    allowed_elements:
        Allowed atomic numbers; ``None`` keeps :data:`REINVENT_DEFAULT_ELEMENTS`.
    neutralize:
        Toggle the charged-form neutralisation pass (step 4).

    Returns
    -------
    The standardised ``Mol``, or ``None`` when the molecule is rejected
    (every rejection reason is logged as a warning -- the toolkit's
    no-silent-drop rule).

    Example
    -------
    >>> from rdkit import Chem
    >>> from pharmaml.chem.standardize import reinvent_standardize_mol
    >>> mol = reinvent_standardize_mol(Chem.MolFromSmiles("CC(=O)[O-].[Na+]"))
    >>> Chem.MolToSmiles(mol)                 # the salt is gone, acid intact
    'CC(=O)O'
    >>> reinvent_standardize_mol(Chem.MolFromSmiles("C")) is None  # too small
    True
    """
    if mol is None:
        return None
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover

    allowed = (
        REINVENT_DEFAULT_ELEMENTS
        if allowed_elements is None
        else frozenset(allowed_elements)
    )

    # 1. largest fragment by heavy-atom count (unsanitised frags on
    # purpose -- some charged inputs only survive the pick; sanitise
    # the winner immediately so the later steps see a healthy Mol)
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:  # noqa: BLE001 - charged hosts fail sanitised splits
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if fragments:
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # noqa: BLE001 - the cleanup step reports it
            _LOG.debug("fragment sanitisation deferred to cleanup: %s", exc)

    # 2. strip explicit hydrogens
    try:
        mol = Chem.RemoveHs(mol)
    except Exception as exc:  # noqa: BLE001 - unsanitisable input falls through
        _LOG.debug("RemoveHs deferred to cleanup: %s", exc)

    # 3. salt removal (RDKit's built-in list)
    try:
        mol = SaltRemover().StripMol(mol, dontRemoveEverything=True)
    except Exception:  # noqa: BLE001 - hostile valences must not kill a batch
        _LOG.warning("salt removal failed on %r -- chain continues",
                     Chem.MolToSmiles(mol))

    # 4. neutralisation to a fixpoint (the replacement mols parse with
    # aromatic-atom warnings outside rings -- silenced like every other
    # RDKit parse in the package)
    if neutralize:
        from rdkit.rdBase import DisableLog, EnableLog

        DisableLog("rdApp.error")
        DisableLog("rdApp.warning")
        try:
            for smarts, replacement in NEUTRALIZE_PATTERNS:
                query = Chem.MolFromSmarts(smarts)
                repl = Chem.MolFromSmiles(replacement)
                if query is None or repl is None:  # pragma: no cover - static table
                    continue
                for _ in range(_MAX_NEUTRALIZE_PASSES):
                    if not mol.HasSubstructMatch(query):
                        break
                    replaced = Chem.ReplaceSubstructs(mol, query, repl)
                    if not replaced:
                        break
                    candidate = replaced[0]
                    if Chem.MolToSmiles(candidate) == Chem.MolToSmiles(mol):
                        break
                    mol = candidate
        finally:
            EnableLog("rdApp.error")
            EnableLog("rdApp.warning")

    # 5. general cleanup (Cleanup mutates IN PLACE in modern RDKit --
    # the REINVENT4 chain predates that API change)
    try:
        Chem.Cleanup(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as exc:  # noqa: BLE001 - the actionable message follows
        _LOG.warning("molecule failed general cleanup: %s", exc)
        return None

    # 6. unwanted patterns reject outright
    for smarts in _UNWANTED_PATTERNS:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            _LOG.warning(
                "molecule rejected: unwanted pattern %s (REINVENT4's "
                "unwanted-pairs list)",
                smarts,
            )
            return None

    # 7. valid size + elements
    n_heavy = mol.GetNumHeavyAtoms()
    if not min_heavy_atoms < n_heavy < max_heavy_atoms:
        _LOG.warning(
            "molecule rejected: %d heavy atoms outside the (%d, %d) window",
            n_heavy,
            min_heavy_atoms,
            max_heavy_atoms,
        )
        return None
    bad_elements = sorted(
        {atom.GetSymbol() for atom in mol.GetAtoms()
         if atom.GetAtomicNum() not in allowed}
    )
    if bad_elements:
        _LOG.warning(
            "molecule rejected: element(s) %s outside the allowed vocabulary",
            bad_elements,
        )
        return None
    return mol


def reinvent_standardize_smiles(
    smiles: str,
    **kwargs: Any,
) -> str | None:
    """:func:`reinvent_standardize_mol` from (and back to) a SMILES.

    The REINVENT4 ``file_reader``/transfer-learning front door: every
    library line standardised through one call, invalid entries dropped
    **with a warning** (never silently).  Returns the canonical SMILES
    of the standardised molecule, or ``None``.

    Example
    -------
    >>> from pharmaml.chem.standardize import reinvent_standardize_smiles
    >>> reinvent_standardize_smiles("CC(=O)[O-].[Na+]")
    'CC(=O)O'
    >>> reinvent_standardize_smiles("[nH]1ccnc1")   # neutralised imidazole
    'c1c[nH]cn1'
    >>> reinvent_standardize_smiles("not-a-smiles") is None
    True
    """
    if not isinstance(smiles, str) or not smiles.strip():
        _LOG.warning("dropping blank SMILES %r", smiles)
        return None
    Chem = require_rdkit_standardize()
    from rdkit.rdBase import DisableLog, EnableLog

    DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001 - hostile strings return None anyway
        mol = None
    finally:
        EnableLog("rdApp.error")
    if mol is None:
        _LOG.warning("dropping unparseable SMILES %r", smiles)
        return None
    standard = reinvent_standardize_mol(mol, **kwargs)
    if standard is None:
        return None
    try:
        return Chem.MolToSmiles(standard)
    except Exception:  # noqa: BLE001 - unsanitisable output drops with a warning
        _LOG.warning("standardised molecule could not be canonicalised")
        return None

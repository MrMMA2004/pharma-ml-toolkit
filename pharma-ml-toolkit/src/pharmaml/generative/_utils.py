"""Private shared helpers for the generative subpackage.

Nothing in this module is public API; it exists so :mod:`.scoring`,
:mod:`.genetic` and :mod:`.rl` share one lazy RDKit gate (mirroring the
``require_rdkit`` precedent in :mod:`pharmaml.chem.descriptors` and the
lazy-torch contract of :mod:`pharmaml.ml.neural`) and one canonicalization
helper with consistent invalid-SMILES handling.
"""

from __future__ import annotations

from pharmaml.exceptions import DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: pip hint embedded in the DependencyError raised when RDKit is missing.
_RDKIT_HINT = (
    "RDKit is required for molecule generation and scoring. "
    "From the toolkit package root, run:  python -m pip install -e '.[chem]'"
)


def require_rdkit():
    """Import and return the RDKit ``Chem`` module or raise helpfully."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(_RDKIT_HINT) from exc
    return Chem


def canonical_smiles(smiles: str) -> str | None:
    """Return the canonical RDKit form of ``smiles`` (``None`` if invalid).

    Blank strings parse as *valid* zero-atom molecules in RDKit, so they
    are treated as invalid here (consistent with the rest of the package).
    """
    Chem = require_rdkit()  # actionable DependencyError when RDKit is absent
# rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
    # but rdBase (their defining module) is fully typed.
    from rdkit.rdBase import DisableLog, EnableLog

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    # RDKit logs parse errors to stderr before returning None; generation
    # loops parse thousands of mutants per run and the raw parser trace is
    # pure noise (same guard family as docking/ligands._validate_smiles).
    DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        EnableLog("rdApp.error")
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_alphabet(smiles_list: list[str]) -> list[str]:
    """Sorted list of the distinct characters used by ``smiles_list``."""
    return sorted({char for smi in smiles_list for char in smi})

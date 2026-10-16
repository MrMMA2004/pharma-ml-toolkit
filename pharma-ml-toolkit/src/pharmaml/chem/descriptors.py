"""Molecular descriptors and fingerprints (RDKit-backed, lazy import).

Upgrades ``others/Ml/Descriptor calculator.py``, ``Morgan_calculator.py``,
``ECFP_calculation.py`` and ``Mordred_calculation.py`` into one module
with:

* a lazy RDKit import that raises an actionable
  :class:`~pharmaml.exceptions.DependencyError`,
* explicit invalid-SMILES accounting (the original silently asserted
  "we are sure all SMILES are valid"),
* reproducible fingerprints via fixed seeds/radii.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["compute_descriptors", "compute_fingerprints", "compute_maccs", "require_rdkit"]

FingerprintKind = Literal["morgan", "rdkit", "atom_pair"]


def require_rdkit():
    """Import and return the RDKit ``Chem`` module or raise helpfully."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for descriptor/fingerprint computation. "
            "Install it with:  pip install 'pharmaml[chem]'  "
            "(or pip install rdkit)"
        ) from exc
    return Chem


def _smiles_to_molecules(
    smiles: pd.Series | list[str],
) -> tuple[list, list[int], list[str]]:
    """Parse a sequence of SMILES, returning (mols, valid_idx, failures)."""
    Chem = require_rdkit()
    molecules, valid_idx, failures = [], [], []
    for idx, smi in enumerate(smiles):
        # RDKit parses "" as a *valid* 0-atom molecule; blank cells must be
        # counted as invalid instead (consistent with
        # utils.validation.validate_smiles_syntax).
        mol = (
            Chem.MolFromSmiles(smi)
            if isinstance(smi, str) and smi.strip()
            else None
        )
        if mol is None:
            failures.append(f"index {idx}: {smi!r}")
        else:
            molecules.append(mol)
            valid_idx.append(idx)
    return molecules, valid_idx, failures


def _as_smiles_list(smiles: Iterable[str] | pd.Series, arg_name: str) -> list[str]:
    """Materialise a SMILES sequence exactly once.

    Rejects a bare string (``"CCO"`` would otherwise be iterated
    character-by-character into three single-atom "molecules") and freezes
    generators / pandas Series so later positional lookups stay valid even
    when the Series index is not ``0..n-1``.
    """
    if isinstance(smiles, str):
        preview = smiles if len(smiles) <= 40 else smiles[:40] + "..."
        raise DataValidationError(
            f"{arg_name} must be a list or pandas.Series of SMILES strings, not a "
            f"single string (got {preview!r}); wrap it in a list to compute "
            "one molecule"
        )
    return list(smiles)


def compute_descriptors(
    smiles: pd.Series | list[str],
    descriptor_set: Literal["rdkit_all", "morgan", "mordred"] = "rdkit_all",
    progress: bool = True,
) -> pd.DataFrame:
    """Compute molecular descriptors for a list of SMILES.

    Parameters
    ----------
    smiles:
        Iterable of SMILES strings (a DataFrame column works directly).
    descriptor_set:
        ``'rdkit_all'`` for all 2D RDKit descriptors, ``'morgan'`` for a
        fixed-radius Morgan count fingerprint, ``'mordred'`` for the
        Mordred descriptor suite (needs ``pip install mordred``).
        Any other value raises instead of silently falling back.
        Known limitation of the Mordred path (documented, deliberately
        not transformed): descriptors that fail for a given molecule
        come back as Mordred ``Error`` objects rather than numbers and
        are kept as-is in the frame -- mordred is exercised only by the
        full-deps CI job, never in the default environment.
    progress:
        Show a tqdm progress bar for large batches.

    Returns
    -------
    pandas.DataFrame
        One row per *valid* molecule, indexed by the original position,
        with a ``SMILES`` column followed by descriptor columns.  Blank /
        whitespace-only SMILES count as invalid, like any other parse
        failure.  Use :func:`compute_fingerprints` for bit/count
        fingerprints instead.

    Raises
    ------
    DependencyError
        If RDKit (or Mordred when selected) is not installed.
    DataValidationError
        If no SMILES parses to a valid molecule, ``smiles`` is a bare
        string, or ``descriptor_set`` is unknown.
    """
    smiles_list = _as_smiles_list(smiles, "smiles")
    molecules, valid_idx, failures = _smiles_to_molecules(smiles_list)
    if failures:
        _LOG.warning("Dropping %d invalid SMILES: %s", len(failures), failures[:5])
    if not molecules:
        raise DataValidationError("No valid SMILES found in the input")

    rows: list[dict] = []
    iterator = molecules
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(molecules, desc="Descriptors", unit="mol")
        except ImportError:
            iterator = molecules

    if descriptor_set == "mordred":
        try:
            from mordred import Calculator, descriptors
        except ImportError as exc:
            raise DependencyError(
                "Mordred is not installed. Install with: pip install mordred"
            ) from exc
        calc = Calculator(descriptors, ignore_3D=True)
        for mol in iterator:
            rows.append(dict(calc(mol)))
        frame = pd.DataFrame(rows)
    elif descriptor_set == "morgan":
        require_rdkit()
        from rdkit.Chem import rdMolDescriptors

        names = [f"FP{i:03d}" for i in range(2048)]
        for mol in iterator:
            counts = rdMolDescriptors.GetHashedMorganFingerprint(
                mol, radius=2, nBits=2048
            )
            # UIntSparseIntVect has GetNonzeroElements(), not GetVal().
            elements = counts.GetNonzeroElements()
            rows.append({name: elements.get(u, 0) for u, name in enumerate(names)})
        frame = pd.DataFrame(rows)
    elif descriptor_set == "rdkit_all":
        from rdkit.Chem import Descriptors

        for mol in iterator:
            rows.append({name: func(mol) for name, func in Descriptors.descList})
        frame = pd.DataFrame(rows)
    else:
        raise DataValidationError(
            "descriptor_set must be one of 'rdkit_all', 'morgan' or 'mordred', "
            f"got {descriptor_set!r}"
        )

    frame.insert(0, "SMILES", [str(smiles_list[i]) for i in valid_idx])
    frame.index = pd.Index(valid_idx, name="original_index")
    _LOG.info("Computed %d descriptors for %d molecules", frame.shape[1] - 1, len(frame))
    return frame


def compute_fingerprints(
    smiles: pd.Series | list[str],
    kind: FingerprintKind = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    as_counts: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Compute fingerprints for a list of SMILES.

    Parameters
    ----------
    smiles:
        Iterable of SMILES strings.
    kind:
        ``'morgan'`` (ECFP-style), ``'rdkit'`` (path-based) or
        ``'atom_pair'`` fingerprint.
    radius:
        Fingerprint radius for Morgan fingerprints.
    n_bits:
        Fold length for hashed fingerprints.
    as_counts:
        Return count fingerprints instead of binary bits (Morgan only).

    Returns
    -------
    tuple
        ``(matrix, valid_indices)`` where ``matrix`` is an
        ``(n_valid, n_bits)`` uint8 (or uint32 for counts) array and
        ``valid_indices`` maps rows back to input positions.  Blank /
        whitespace-only SMILES count as invalid, like any other parse
        failure.

    Raises
    ------
    DependencyError
        If RDKit is not installed.
    DataValidationError
        If ``smiles`` is a bare string, ``radius`` / ``n_bits`` are out
        of range, or ``as_counts=True`` is combined with a ``kind``
        other than ``'morgan'`` (the count flavour only exists for
        Morgan fingerprints -- other kinds would silently return
        binary bits instead).
    """
    if radius < 0:
        raise DataValidationError(f"radius must be >= 0, got {radius}")
    if not 0 < n_bits <= 65536:
        raise DataValidationError(f"n_bits out of range: {n_bits}")
    if as_counts and kind != "morgan":
        raise DataValidationError(
            "as_counts=True is only supported for kind='morgan' (count "
            f"fingerprints); got kind={kind!r} -- pass as_counts=False or "
            "switch to morgan"
        )

    smiles_list = _as_smiles_list(smiles, "smiles")
    require_rdkit()
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    molecules, valid_idx, failures = _smiles_to_molecules(smiles_list)
    if failures:
        _LOG.warning("Dropping %d invalid SMILES: %s", len(failures), failures[:5])

    if kind == "morgan":
        if as_counts:
            fps = [
                rdMolDescriptors.GetHashedMorganFingerprint(m, radius=radius, nBits=n_bits)
                for m in molecules
            ]
            matrix = np.zeros((len(fps), n_bits), dtype=np.uint32)
            for row, fp in enumerate(fps):
                for idx, value in fp.GetNonzeroElements().items():
                    matrix[row, idx] = value
            return matrix, valid_idx
        fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            for m in molecules
        ]
    elif kind == "rdkit":
        from rdkit.Chem import RDKFingerprint

        fps = [RDKFingerprint(m, fpSize=n_bits) for m in molecules]
    elif kind == "atom_pair":
        fps = [
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits)
            for m in molecules
        ]
    else:
        raise DataValidationError(f"Unknown fingerprint kind {kind!r}")

    matrix = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for row, fp in enumerate(fps):
        buffer = np.zeros((n_bits,), dtype=np.uint8)
        # DataStructs.ConvertToNumpyArray fills an existing 1-D array.
        DataStructs.ConvertToNumpyArray(fp, buffer)
        matrix[row] = buffer
    _LOG.info("Computed %d x %d %s fingerprints", *matrix.shape, kind)
    return matrix, valid_idx


# Re-exported for callers that want the MACCS keys shortcut.
def compute_maccs(smiles: pd.Series | list[str]) -> tuple[np.ndarray, list[int]]:
    """Compute the 167-bit MACCS key fingerprints.

    Blank / whitespace-only SMILES count as invalid and are dropped with
    a warning naming their input index, matching the other entry points.

    Raises
    ------
    DependencyError
        If RDKit is not installed.
    DataValidationError
        If ``smiles`` is a bare string.
    """
    require_rdkit()  # import guard
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    molecules, valid_idx, failures = _smiles_to_molecules(
        _as_smiles_list(smiles, "smiles")
    )
    if failures:
        _LOG.warning("Dropping %d invalid SMILES: %s", len(failures), failures[:5])
    n_bits = 167
    matrix = np.zeros((len(molecules), n_bits), dtype=np.uint8)
    for row, mol in enumerate(molecules):
        # MACCSkeys.GenMACCSKeys is a re-export of this rdMolDescriptors
        # entry point (the same object at runtime) -- the defining module
        # is the one rdkit-stubs actually types.
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr := np.zeros((n_bits,), dtype=np.uint8))
        matrix[row] = arr
    return matrix, valid_idx

"""Molecular descriptors and fingerprints (RDKit-backed, lazy import).

Upgrades ``others/Ml/Descriptor calculator.py``, ``Morgan_calculator.py``,
``ECFP_calculation.py``, ``Mordred_calculation.py`` and ``xxx.ipynb``
cell 6's 3D block into one module with:

* a lazy RDKit import that raises an actionable
  :class:`~pharmaml.exceptions.DependencyError`,
* explicit invalid-SMILES accounting (the original silently asserted
  "we are sure all SMILES are valid"),
* reproducible fingerprints via fixed seeds/radii,
* reproducible 3D shape descriptors (ETKDGv3 at the fixed seed 42,
  MMFF94s relaxation with the UFF fallback).

Examples
--------
>>> from pharmaml.chem.descriptors import compute_descriptors
>>> frame = compute_descriptors(["CC(=O)Oc1ccccc1C(=O)O"], progress=False)  # aspirin
>>> round(float(frame["MolWt"].iloc[0]), 2)          # exact RDKit MolWt
180.16
>>> round(float(frame["MolLogP"].iloc[0]), 4)        # Crippen LogP
1.3101
>>> round(float(frame["TPSA"].iloc[0]), 1)           # topological polar area
63.6
>>> int(frame["NumHDonors"].iloc[0]), int(frame["NumHAcceptors"].iloc[0])  # HBD/HBA
(1, 3)

>>> frame = compute_descriptors(["CCO", "not-a-smiles", "CCC"], progress=False)
>>> list(frame.index), frame["SMILES"].tolist()      # invalid row dropped
([0, 2], ['CCO', 'CCC'])
>>> compute_descriptors(["!!!"], progress=False)     # every row invalid -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: No valid SMILES found in the input
>>> compute_descriptors([], progress=False)          # empty input -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smiles is empty -- pass at least one SMILES string

>>> frame = compute_descriptors(["CCO", "[H]", "CCC"], progress=False)
>>> list(frame.index), frame["SMILES"].tolist()      # zero-heavy-atom [H] dropped
([0, 2], ['CCO', 'CCC'])
>>> compute_descriptors(["[H]"], progress=False)    # every row failed -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: Descriptor computation failed for every valid molecule

>>> from pharmaml.chem.descriptors import compute_fingerprints, compute_maccs
>>> from pharmaml.chem.similarity import tanimoto_from_bitarrays
>>> matrix, valid = compute_fingerprints(["CCO", "CCO", "C"], n_bits=64)
>>> matrix.shape, matrix.dtype.name, valid
((3, 64), 'uint8', [0, 1, 2])
>>> tanimoto_from_bitarrays(matrix[0], matrix[1])    # identical molecules -> 1
1.0
>>> tanimoto_from_bitarrays(matrix[0], matrix[2])    # ethanol vs methane: no shared bit
0.0
>>> maccs, _ = compute_maccs(["CCO"])
>>> maccs.shape, int(maccs.sum())                    # 167-bit MACCS keys
((1, 167), 9)
>>> compute_fingerprints(["CCO"], radius=-1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: radius must be >= 0, got -1
>>> compute_fingerprints(["CCO"], radius="2")        # non-int radius -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: radius must be an integer >= 0, got '2'

>>> from pharmaml.chem.descriptors import D3D_COLUMNS, compute_3d_descriptors
>>> len(D3D_COLUMNS)                             # the 3D shape descriptor contract
9
>>> frame = compute_3d_descriptors(["CC(=O)Oc1ccccc1C(=O)O"], progress=False)  # aspirin
>>> list(frame.columns) == ["SMILES", *D3D_COLUMNS]
True
>>> round(float(frame["PMI1"].iloc[0]), 2), round(float(frame["RadiusOfGyration"].iloc[0]), 3)
(463.52, 2.428)
>>> row = frame[list(D3D_COLUMNS)].iloc[0].to_numpy()
>>> bool(row[2] <= row[3] <= row[4])            # PMI1 <= PMI2 <= PMI3
True
>>> twin = compute_3d_descriptors(["CC(=O)Oc1ccccc1C(=O)O"], progress=False)
>>> float(twin["PMI1"].iloc[0]) == float(frame["PMI1"].iloc[0])   # deterministic
True
>>> frame = compute_3d_descriptors(["CCO", "not-a-smiles", "CCC"], progress=False)
>>> list(frame.index), frame["SMILES"].tolist()   # invalid row dropped, valid rows clean
([0, 2], ['CCO', 'CCC'])
>>> compute_3d_descriptors("CCO", progress=False)  # bare string -> typed error
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: smiles must be a list or pandas.Series of SMILES strings, not a single string (got 'CCO'); wrap it in a list to compute one molecule
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "D3D_COLUMNS",
    "compute_3d_descriptors",
    "compute_descriptors",
    "compute_fingerprints",
    "compute_maccs",
    "require_rdkit",
]

FingerprintKind = Literal["morgan", "rdkit", "atom_pair", "avalon"]

#: The nine RDKit 3D shape descriptor columns, in output order (after
#: ``SMILES``) -- the RDKit half of ``xxx.ipynb`` cell 6's 3D block
#: (the cell's mordred WHIM/RDF half stays with the mordred
#: ``descriptor_set`` of :func:`compute_descriptors`).
D3D_COLUMNS: tuple[str, ...] = (
    "Asphericity",
    "Eccentricity",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
    "NPR1",
    "NPR2",
)


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
        For ``'mordred'``, failed descriptor cells are part of the
        answer, not exceptions: mordred reports them as ``Missing``
        (known-uncomputable for this molecule, e.g. ring descriptors
        on an acyclic input) or ``Error`` (unexpected failure) value
        objects, which are converted to NaN here -- one aggregate
        warning names the cell counts and a few example descriptor
        names.  Column names are mordred's own string descriptor
        names (``'MW'``, ``'nH'``, ...).  An all-NaN descriptor column
        is a real mordred verdict, not a bug (downstream, the ML
        pipeline's SimpleImputer drops all-NaN columns silently) --
        no fillna or imputation happens beyond the
        Error/Missing-to-NaN conversion.
    progress:
        Show a tqdm progress bar for large batches.

    Returns
    -------
    pandas.DataFrame
        One row per *valid* molecule, indexed by the original position,
        with a ``SMILES`` column followed by descriptor columns.  Blank /
        whitespace-only SMILES count as invalid, like any other parse
        failure.  For ``'rdkit_all'``, a molecule whose descriptor
        computation raises is dropped the same way, with a warning naming
        its original index -- e.g. zero-heavy-atom species such as ``[H]``
        (RDKit's SPS descriptor divides by the heavy-atom count).  For
        ``'mordred'``, the column names are mordred's string descriptor
        names and failed cells (``Error``/``Missing`` value objects)
        arrive as NaN -- the aggregate-warning contract above.  Use
        :func:`compute_fingerprints` for bit/count fingerprints instead.

    Raises
    ------
    DependencyError
        If RDKit (or Mordred when selected) is not installed.
    DataValidationError
        If ``smiles`` is a bare string or empty, ``descriptor_set`` is
        unknown, no SMILES parses to a valid molecule, or (for
        ``'rdkit_all'``) descriptor computation fails for every valid
        molecule.
    """
    smiles_list = _as_smiles_list(smiles, "smiles")
    # Cheap validation BEFORE the parse loop (and before any progress-bar
    # spin-up): an empty input is a caller bug rather than a parse outcome,
    # and an unknown descriptor_set must not wait for a full parse to
    # surface (error types unchanged -- only ordering and message quality).
    if not smiles_list:
        raise DataValidationError("smiles is empty -- pass at least one SMILES string")
    if descriptor_set not in ("rdkit_all", "morgan", "mordred"):
        raise DataValidationError(
            "descriptor_set must be one of 'rdkit_all', 'morgan' or 'mordred', "
            f"got {descriptor_set!r}"
        )
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
            from mordred.error import Error, Missing
        except ImportError as exc:
            raise DependencyError(
                "Mordred is not installed. Install with: pip install mordred"
            ) from exc
        calc = Calculator(descriptors, ignore_3D=True)
        # mordred's Result object is NOT a dict: keys() yields Descriptor
        # *instances* and failed cells are Error/Missing sentinel objects
        # (both convert to NaN via __float__).  Building the frame from
        # dict(calc(mol)) therefore produced callable object column names
        # AND object-dtype cells -- unusable downstream (100x-52).  The
        # official asdict()/fill_missing() pair gives string keys and NaN
        # cells; the counting pass below feeds the aggregate warning.
        error_cells = 0
        missing_cells = 0
        example_names: list[str] = []
        for mol in iterator:
            result = calc(mol)
            for desc, value in result.items():
                if isinstance(value, (Error, Missing)):
                    if isinstance(value, Error):
                        error_cells += 1
                    else:
                        missing_cells += 1
                    if len(example_names) < 3:
                        example_names.append(str(desc))
            rows.append(result.fill_missing().asdict())
        frame = pd.DataFrame(rows)
        if error_cells or missing_cells:
            total_cells = len(rows) * len(calc.descriptors)
            _LOG.warning(
                "Mordred could not compute %d of %d descriptor cells "
                "(%d Error, %d Missing; e.g. %s) -- kept as NaN in the frame",
                error_cells + missing_cells,
                total_cells,
                error_cells,
                missing_cells,
                ", ".join(example_names),
            )
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

        # Per-molecule guard: some RDKit descriptors divide by the heavy-atom
        # count (SPS), which is zero for species like "[H]"/"[H][H]" -- a
        # single such molecule used to crash the whole batch with a raw
        # ZeroDivisionError.  Drop the offending row with a warning naming
        # its original index (the invalid-SMILES accounting style) and keep
        # valid_idx aligned with the surviving rows.
        kept_idx: list[int] = []
        for orig_idx, mol in zip(valid_idx, iterator):
            try:
                rows.append({name: func(mol) for name, func in Descriptors.descList})
            except Exception as exc:  # noqa: BLE001 — RDKit raises a zoo per molecule
                _LOG.warning(
                    "Dropping molecule at index %d (%r): descriptor computation "
                    "failed (%s: %s)",
                    orig_idx,
                    smiles_list[orig_idx],
                    type(exc).__name__,
                    exc,
                )
                continue
            kept_idx.append(orig_idx)
        valid_idx = kept_idx
        if not rows:
            raise DataValidationError(
                "Descriptor computation failed for every valid molecule"
            )
        frame = pd.DataFrame(rows)
    # No else: descriptor_set membership is validated before parsing above.

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
        ``'morgan'`` (ECFP-style), ``'rdkit'`` (path-based),
        ``'atom_pair'`` or ``'avalon'`` fingerprint.  Avalon upgrades
        the rar's ``Descriptor classification.py`` / ``Descriptor
        regression.py`` feature unions, which folded 512-bit Avalon
        count fingerprints alongside ECFP/RDKit/Morgan blocks.
    radius:
        Fingerprint radius for Morgan fingerprints.  Must be an integer
        (numpy integers are accepted); bools are rejected -- see Raises.
    n_bits:
        Fold length for hashed fingerprints.  Must be an integer in
        ``(0, 65536]`` (numpy integers are accepted); bools are rejected.
    as_counts:
        Return count fingerprints instead of binary bits (Morgan and
        Avalon -- the only two kinds with a count flavour; the rar's
        Avalon block used counts).

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
        If ``smiles`` is a bare string, ``radius`` / ``n_bits`` are not
        integers or are out of range, or ``as_counts=True`` is combined
        with a ``kind`` that has no count flavour (only ``'morgan'`` and
        ``'avalon'`` do -- other kinds would silently return binary
        bits instead).  bools count as non-integers here by
        decision: ``True``/``False`` are nonsense as a radius or bit
        count, so they are rejected instead of being silently read as
        ``1``/``0``.
    """
    # Int-ness BEFORE the range checks: str/None leak a raw TypeError from
    # the comparisons below, and floats pass them only to blow up inside
    # the RDKit bindings with a Boost.Python ArgumentError (C++ signature
    # dump).  bools are int subclasses but are rejected explicitly (see
    # docstring); numpy integers are accepted and normalised to plain int
    # (the Boost bindings reject numpy scalar types).
    if not isinstance(radius, (int, np.integer)) or isinstance(radius, bool):
        raise DataValidationError(f"radius must be an integer >= 0, got {radius!r}")
    radius = int(radius)
    if radius < 0:
        raise DataValidationError(f"radius must be >= 0, got {radius}")
    if not isinstance(n_bits, (int, np.integer)) or isinstance(n_bits, bool):
        raise DataValidationError(
            f"n_bits must be an integer in (0, 65536], got {n_bits!r}"
        )
    n_bits = int(n_bits)
    if not 0 < n_bits <= 65536:
        raise DataValidationError(f"n_bits out of range: {n_bits}")
    if as_counts and kind not in {"morgan", "avalon"}:
        raise DataValidationError(
            "as_counts=True is only supported for count-capable kinds "
            f"('morgan', 'avalon'); got kind={kind!r} -- pass "
            "as_counts=False or switch to morgan/avalon"
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
    elif kind == "avalon":
        from rdkit.Avalon import pyAvalonTools

        if as_counts:
            # GetAvalonCountFP hashes straight into nBits (the rar's
            # Descriptor scripts folded the default-512 sparse vector
            # by hand -- the binding does it natively).
            fps = [
                pyAvalonTools.GetAvalonCountFP(m, nBits=n_bits) for m in molecules
            ]
            matrix = np.zeros((len(fps), n_bits), dtype=np.uint32)
            for row, fp in enumerate(fps):
                for idx, value in fp.GetNonzeroElements().items():
                    matrix[row, idx] = value
            _LOG.info("Computed %d x %d avalon count fingerprints", *matrix.shape)
            return matrix, valid_idx
        fps = [pyAvalonTools.GetAvalonFP(m, nBits=n_bits) for m in molecules]
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


def _shape_vector(mol_h: Any, n_conformers: int) -> np.ndarray:
    """Embed, optimise and describe one hydrogenated molecule in 3D.

    The ``xxx.ipynb`` cell 6 pipeline: ETKDGv3 embedding at the fixed
    seed 42 with ``n_conformers`` conformers, MMFF94s relaxation (200
    iterations) with the UFF fallback the notebook carries, then the
    nine :data:`D3D_COLUMNS` shape descriptors read off the
    lowest-energy successfully optimised conformer.  The molecule must
    already carry explicit hydrogens (``Chem.AddHs``).

    Raises
    ------
    ValueError
        When no conformer embeds, or neither force field can optimise
        any conformer to a finite energy -- the caller decides the
        failure policy (``compute_3d_descriptors`` keeps a NaN row).
    """
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolDescriptors

    # rdDistGeom/rdForceFieldHelpers instead of the AllChem facade: the
    # AllChem re-export module is what rdkit-stubs leaves untyped (the
    # rism module's own pin -- same embedding calls, same stubs).
    params = rdDistGeom.ETKDGv3()
    # rdkit-stubs types randomSeed as EmbedParameters (a stub bug); the
    # runtime attribute is the integer RNG seed (the rism module's pin).
    params.randomSeed = 42  # type: ignore[assignment]
    conf_ids = list(
        rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    )
    if not conf_ids:
        raise ValueError("ETKDGv3 embedding produced no conformers")

    try:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_h, maxIters=200, mmffVariant="MMFF94s"
        )
        # Flag -1 means the MMFF typing failed for that conformer (RDKit
        # reports it as (-1, -1.0)) -- the sanity failure that triggers
        # the notebook's UFF fallback, here detected explicitly instead
        # of through the exception path alone.
        mmff_failed = any(flag < 0 for flag, _energy in results)
    except Exception:  # noqa: BLE001 — RDKit raises a zoo per molecule
        mmff_failed = True
        results = []
    if mmff_failed:
        try:
            results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol_h, maxIters=200)
        except Exception as exc:
            raise ValueError(
                f"neither MMFF nor UFF could optimise the molecule ({type(exc).__name__})"
            ) from exc

    # Successfully optimised conformers: flag 0 (converged) or 1 (hit the
    # iteration cap but still carries a valid optimised geometry+energy);
    # only -1 (typing failure) is unusable.  The notebook filtered to
    # flag == 0 alone, which empties the list whenever every conformer
    # hits the cap and silently NaNs an optimisable molecule.
    usable = [
        (position, float(energy))
        for position, (flag, energy) in enumerate(results)
        if flag >= 0 and np.isfinite(energy)
    ]
    if not usable:
        raise ValueError("no conformer optimised with a finite energy")
    best_position = min(usable, key=lambda pair: pair[1])[0]
    # conf_ids is aligned with the conformer-list order (a freshly
    # hydrogenated molecule numbers its conformers 0..n-1), mirroring
    # the notebook's GetConformer(id=argmin) access.
    conf_id = conf_ids[best_position]
    # The calculator order matches D3D_COLUMNS exactly (the notebook's
    # own dict literal, spelled as direct calls so rdkit-stubs types it).
    return np.asarray(
        [
            rdMolDescriptors.CalcAsphericity(mol_h, confId=conf_id),
            rdMolDescriptors.CalcEccentricity(mol_h, confId=conf_id),
            rdMolDescriptors.CalcPMI1(mol_h, confId=conf_id),
            rdMolDescriptors.CalcPMI2(mol_h, confId=conf_id),
            rdMolDescriptors.CalcPMI3(mol_h, confId=conf_id),
            rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=conf_id),
            rdMolDescriptors.CalcSpherocityIndex(mol_h, confId=conf_id),
            rdMolDescriptors.CalcNPR1(mol_h, confId=conf_id),
            rdMolDescriptors.CalcNPR2(mol_h, confId=conf_id),
        ],
        dtype=float,
    )


def compute_3d_descriptors(
    smiles: pd.Series | list[str],
    n_conformers: int = 3,
    progress: bool = True,
) -> pd.DataFrame:
    """Compute RDKit 3D shape descriptors for a list of SMILES.

    The library port of ``xxx.ipynb`` cell 6's RDKit half: for every
    valid molecule the hydrogenated structure is embedded with ETKDGv3
    (fixed seed 42, ``n_conformers`` conformers), relaxed with MMFF94s
    (200 iterations, UFF fallback -- the notebook's own fallback, also
    triggered by MMFF typing failures), and the nine
    :data:`D3D_COLUMNS` shape descriptors are read off the
    lowest-energy successfully optimised conformer.

    Parameters
    ----------
    smiles:
        Iterable of SMILES strings (a DataFrame column works directly).
        A bare string is rejected -- wrap it in a list.
    n_conformers:
        Number of ETKDG conformers per molecule (the notebook's default
        3).  Must be an integer >= 1; bools are rejected like the other
        integer parameters of this module.
    progress:
        Show a tqdm progress bar for large batches.

    Returns
    -------
    pandas.DataFrame
        One row per *valid* molecule, indexed by the original position
        (index name ``original_index``), with a ``SMILES`` column
        followed by the nine :data:`D3D_COLUMNS` columns (float).  A
        molecule whose embedding or optimisation fails keeps its row
        with all-NaN descriptor cells -- the notebook's
        failed-molecule behaviour -- instead of being dropped; invalid
        SMILES are dropped with a warning naming their input index,
        exactly like :func:`compute_descriptors`.

    Raises
    ------
    DependencyError
        If RDKit is not installed.
    DataValidationError
        If ``smiles`` is a bare string or empty, ``n_conformers`` is
        not an integer >= 1, or no SMILES parses to a valid molecule.

    Notes
    -----
    Determinism: the ETKDGv3 embedding uses the fixed seed 42 and the
    optimiser is deterministic, so repeated calls on the same input
    return identical descriptor values.  Deviations from the notebook,
    both deliberate: (1) the lowest-energy pick considers conformers
    that hit the iteration cap (MMFF flag 1) as well as fully converged
    ones -- the notebook's ``flag == 0`` filter empties on
    capped-but-optimised molecules; (2) the mordred WHIM/RDF half of
    the cell stays with the ``'mordred'`` ``descriptor_set`` of
    :func:`compute_descriptors` rather than being silently dropped.
    """
    smiles_list = _as_smiles_list(smiles, "smiles")
    if not smiles_list:
        raise DataValidationError("smiles is empty -- pass at least one SMILES string")
    if (
        not isinstance(n_conformers, (int, np.integer))
        or isinstance(n_conformers, bool)
        or int(n_conformers) < 1
    ):
        raise DataValidationError(
            f"n_conformers must be an integer >= 1, got {n_conformers!r}"
        )
    n_conformers = int(n_conformers)

    Chem = require_rdkit()
    molecules, valid_idx, failures = _smiles_to_molecules(smiles_list)
    if failures:
        _LOG.warning("Dropping %d invalid SMILES: %s", len(failures), failures[:5])
    if not molecules:
        raise DataValidationError("No valid SMILES found in the input")

    iterator = molecules
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(molecules, desc="3D descriptors", unit="mol")
        except ImportError:
            iterator = molecules

    rows: list[dict[str, float]] = []
    n_failed = 0
    for orig_idx, mol in zip(valid_idx, iterator):
        try:
            values = _shape_vector(Chem.AddHs(mol), n_conformers)
            rows.append(dict(zip(D3D_COLUMNS, map(float, values))))
        except Exception as exc:  # noqa: BLE001 — RDKit raises a zoo per molecule
            n_failed += 1
            _LOG.warning(
                "3D descriptor computation failed for molecule at index %d (%r) "
                "(%s: %s) -- kept as a NaN row",
                orig_idx,
                smiles_list[orig_idx],
                type(exc).__name__,
                exc,
            )
            rows.append({name: float("nan") for name in D3D_COLUMNS})

    frame = pd.DataFrame(rows, columns=list(D3D_COLUMNS), dtype=float)
    frame.insert(0, "SMILES", [str(smiles_list[i]) for i in valid_idx])
    frame.index = pd.Index(valid_idx, name="original_index")
    if rows and n_failed == len(rows):
        _LOG.warning(
            "3D descriptor computation failed for every valid molecule -- "
            "the frame is all NaN"
        )
    _LOG.info(
        "Computed %d 3D shape descriptors for %d molecules (%d NaN row(s), "
        "n_conformers=%d)",
        len(D3D_COLUMNS),
        len(frame),
        n_failed,
        n_conformers,
    )
    return frame


def _shape_computer(n_conformers: int = 3) -> Callable[[Any], np.ndarray]:
    """Feature-factory computer: mol -> nine-value 3D shape vector.

    The molecule must carry only implicit hydrogens (the feature
    factory parses with ``Chem.MolFromSmiles`` and hydrogenates here).
    A molecule that cannot be embedded or optimised yields an all-NaN
    vector -- the family keeps the ROW (the notebook's behaviour)
    instead of dropping the molecule.
    """

    def shape_vector(mol: Any) -> np.ndarray:
        from rdkit import Chem

        try:
            return _shape_vector(Chem.AddHs(mol), n_conformers)
        except Exception:  # noqa: BLE001 — RDKit raises a zoo per molecule
            return np.full(len(D3D_COLUMNS), np.nan)

    return shape_vector

"""Multi-family molecular feature factory: compute, rank, select.

Upgrades the uploaded ``Descriptor classification.py`` /
``Descriptor regression.py`` scripts into a tested library module.  The
originals computed 7 feature families side by side (~10,800 columns per
molecule), ranked them with a fitted model, then kept the top-K features
through a two-layer selection; this module keeps that pipeline but replaces
the notebook glue with explicit contracts:

* **One vocabulary.**  :data:`FEATURE_FAMILIES` fixes the canonical family
  order and :data:`FAMILY_PREFIXES` the column naming, so a feature frame
  is always ``desc_*`` + ``apc_bit_*`` + ``ecfp_bit_*`` +
  ``ecfp3_bit_*`` + ``avalon_count_*`` + ``rdkit_bit_*`` +
  ``morgan_full_bit_*`` + ``morgan_scaffold_bit_*`` + the ten
  ``fak_ml.ipynb`` fingerprint families (``maccs_bit_*``, ``fcfp_bit_*``,
  ``fcfp_count_*``, ``topo_bit_*``, ``topo_count_*``, ``rdkit_count_*``,
  ``pattern_bit_*``, ``layered_bit_*``, ``estate_bit_*``, ``erg_*``) +
  ``d3d_*`` + ``rism_*`` (thirteen 2048-bit blocks + 512 + 167 + 79 +
  315 + the current RDKit property list + the nine 3D shape descriptors
  + the ten 3D-RISM-KH solvation descriptors -- more than 27,000 columns
  in total).  The ``ecfp3`` family (radius-3 Morgan, the
  ``Ml/D2.py`` upload's own R3 half of its dual-radius fingerprints)
  and the ten ``fak_ml.ipynb`` families (cell 40's 16 datamol
  fingerprint types minus the six the vocabulary already carried) join
  as first-class citizens; the ``shape_3d`` family (``xxx.ipynb`` cell
  6's 3D block: ETKDGv3 conformers, MMFF94s->UFF fallback, 9
  rdMolDescriptors shape descriptors) and ``rism`` close the frame as
  the two 3D-geometry descriptor families.
* **Failure accounting instead of silent drops.**  Molecules that fail to
  parse or compute are dropped from the frame and recorded with their
  original row index in :class:`FeatureReport`; the Murcko-scaffold
  family deliberately keeps zero rows (an acyclic molecule has an *empty*
  scaffold, which is a zero fingerprint, not a failure), and the
  ``shape_3d`` family keeps NaN rows for molecules whose 3D pipeline
  fails (``xxx.ipynb`` cell 6's failed-molecule behaviour -- the row
  stays, the descriptors do not).
* **Modern RDKit calls.**  The atom-pair count fingerprint comes from
  ``rdFingerprintGenerator`` sparse counts folded modulo 2048 (the
  deprecated ``GetAtomPairFingerprint`` and the rar's datamol dependency
  are both gone); the Murcko scaffold uses
  ``MurckoScaffold.GetScaffoldForMol`` (the rar's ``GetMurckoScaffold``
  no longer exists in RDKit 2026.09); the ten ``fak_ml.ipynb`` families
  reproduce datamol's own ``fp.py`` mapping exactly, with three honest
  build notes: (1) datamol's 'topological' name maps to
  ``GetTopologicalTorsionGenerator`` (the 4-atom torsion fingerprint
  with ``countSimulation=True`` -- NOT the RDKit path fingerprint, a
  first draft here confused the two; the torsion bits are verified
  byte-exact against ``dm.to_fp(fp_type='topological')``);
  (2) datamol's 'rdkit-count' carries ``numBitsPerFeature=1`` while the
  bit flavour 'rdkit' uses 2 -- the count family must halve its naive
  generator counts to match (aspirin: 301 features, not the 602 a
  default generator double-counts); (3) the generator objects expose no
  ``GetCountFingerprintAsBitVect`` on RDKit 2026.09, so the folded count
  vectors arrive as ``GetCountFingerprintAsNumPy``.
* **Strictly optional LightGBM.**  :func:`rank_features` imports lightgbm
  lazily and falls back to random forests when it is not installed, so
  this module's import path needs only rdkit + numpy + pandas
  (scikit-learn is imported inside the ranking functions as well).
* **Two-layer selection with stable ties.**  Layer 1 keeps the columns at
  or above their family's median importance (the rar's per-family
  ``SelectFromModel(threshold='median')``); layer 2 takes the top-K by
  score, ties broken by frame column order for reproducibility.

Examples
--------
>>> from pharmaml.chem.features import (
...     compute_feature_frame, family_of_column, rank_features,
...     select_top_features, selection_metadata)

>>> frame, report = compute_feature_frame(["CCO"], families=("descriptors",))
>>> (frame.shape[0], round(float(frame["desc_amw"].iloc[0]), 2))  # RDKit amw
(1, 46.07)
>>> report.smiles, report.dropped, report.n_valid
(['CCO'], [], 1)

>>> frame, report = compute_feature_frame(["CCO"], families=("ecfp", "avalon_count"))
>>> frame.shape, report.total_columns
((1, 2560), 2560)
>>> (frame.columns[0], frame.columns[2048], frame.columns[-1])
('ecfp_bit_0', 'avalon_count_0', 'avalon_count_511')

>>> frame, report = compute_feature_frame(
...     ["CCO", "not_a_smiles", "CCC"], families=("ecfp",))
>>> list(frame.index), report.smiles      # invalid row dropped + recorded
([0, 2], ['CCO', 'CCC'])
>>> report.dropped
[(1, 'not_a_smiles')]

>>> compute_feature_frame([], families=("ecfp",))
Traceback (most recent call last):
...
ValueError: smiles is empty -- pass at least one SMILES string
>>> compute_feature_frame(["CCO"], families=("ecfp", "typo"))
Traceback (most recent call last):
...
ValueError: unknown feature family 'typo'; valid families are: descriptors, atompair_count, ecfp, ecfp3, avalon_count, rdkit_fingerprint, morgan_full, morgan_scaffold, maccs, fcfp, fcfp_count, topological, topological_count, rdkit_count, pattern, layered, estate, erg, shape_3d, rism
>>> compute_feature_frame(["CCO"], families=())
Traceback (most recent call last):
...
ValueError: families is empty -- pass at least one family name

>>> (family_of_column("desc_exactmw"), family_of_column("ecfp_bit_17"))
('descriptors', 'ecfp')
>>> family_of_column("ecfp3_bit_2047")
'ecfp3'
>>> family_of_column("maccs_bit_166")
'maccs'
>>> family_of_column("d3d_pmi1")
'shape_3d'
>>> family_of_column("SMILES")
Traceback (most recent call last):
...
ValueError: unknown feature-column prefix in 'SMILES'; columns must start with one of: desc_, apc_bit_, ecfp_bit_, ecfp3_bit_, avalon_count_, rdkit_bit_, morgan_full_bit_, morgan_scaffold_bit_, maccs_bit_, fcfp_bit_, fcfp_count_, topo_bit_, topo_count_, rdkit_count_, pattern_bit_, layered_bit_, estate_bit_, erg_, d3d_, rism_

>>> frame, report = compute_feature_frame(
...     ["CCO"], families=("maccs", "estate", "erg", "shape_3d"))
>>> report.families                     # the fak_ml widths on this build
{'maccs': 167, 'estate': 79, 'erg': 315, 'shape_3d': 9}
>>> (frame.columns[0], frame.columns[-1])
('maccs_bit_0', 'd3d_npr2')
>>> float(frame["d3d_pmi1"].iloc[0]) > 0.0    # ethanol embeds & optimises
True

>>> frame, report = compute_feature_frame(["CCCCCCCC"], families=("morgan_scaffold",))
>>> (frame.shape, float(frame.iloc[0].sum()), report.n_valid)   # acyclic: kept
((1, 2048), 0.0, 1)

>>> import numpy as np, pandas as pd
>>> variance_frame = pd.DataFrame({
...     "f_informative": [0, 1, 2, 3, 4],
...     "f_noise": [1, -1, 1, -1, 1],
...     "f_constant": [5, 5, 5, 5, 5]})
>>> scores = rank_features(variance_frame)
>>> (scores.index.tolist(), float(scores["f_informative"]), float(scores["f_constant"]))
(['f_informative', 'f_noise', 'f_constant'], 2.5, 0.0)

>>> x = np.arange(20, dtype=float)
>>> model_frame = pd.DataFrame({
...     "f_informative": x,
...     "f_noise": np.tile([1.0, -1.0], 10),
...     "f_constant": np.full(20, 7.0)})
>>> scores = rank_features(model_frame, 3.0 * x, method="random_forest")
>>> (scores.index[0], float(scores["f_constant"]))
('f_informative', 0.0)

>>> selection_frame = pd.DataFrame(np.zeros((3, 6)), columns=[
...     "desc_a", "desc_b", "desc_c", "ecfp_bit_0", "ecfp_bit_1",
...     "morgan_full_bit_0"])
>>> scores = pd.Series({"desc_a": 0.10, "desc_b": 0.50, "desc_c": 0.30,
...                     "ecfp_bit_0": 0.40, "ecfp_bit_1": 0.20,
...                     "morgan_full_bit_0": 0.35})
>>> selection = select_top_features(selection_frame, scores, k=3, method="demo")
>>> selection.selected
['desc_b', 'ecfp_bit_0', 'morgan_full_bit_0']
>>> selection.family_counts
{'descriptors': 1, 'ecfp': 1, 'morgan_full': 1}

>>> frame, report = compute_feature_frame(
...     ["CCO", "not_a_smiles"], families=("ecfp",))
>>> scores = pd.Series(1.0, index=frame.columns)
>>> selection = select_top_features(frame, scores, k=2)
>>> meta = selection_metadata(selection, report, k=2)
>>> sorted(meta)
['descriptor_names', 'dropped', 'families_computed', 'family_counts', 'fingerprint_sizes', 'k_requested', 'method', 'n_molecules', 'selected_features', 'total_features']
>>> (meta["dropped"], meta["n_molecules"], meta["fingerprint_sizes"])
([[1, 'not_a_smiles']], 1, {'ecfp': 2048})
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.descriptors import require_rdkit
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "ATOMPAIR_BITS",
    "AVALON_BITS",
    "D3D_DESCRIPTOR_COUNT",
    "ECFP3_BITS",
    "ECFP3_RADIUS",
    "ECFP_BITS",
    "ERG_BITS",
    "ESTATE_BITS",
    "FAMILY_PREFIXES",
    "FCFP_BITS",
    "FEATURE_FAMILIES",
    "LAYERED_BITS",
    "MACCS_BITS",
    "MORGAN_BITS",
    "MORGAN_RADIUS",
    "PATTERN_BITS",
    "RDKIT_FP_BITS",
    "RISM_DESCRIPTOR_COUNT",
    "TOPOLOGICAL_BITS",
    "FeatureReport",
    "FeatureSelection",
    "VarianceFilterResult",
    "compute_feature_frame",
    "family_of_column",
    "filter_low_variance",
    "rank_features",
    "select_top_features",
    "selection_metadata",
]

#: The twenty feature families, in canonical output order (the two 3D
#: geometry descriptor families, ``shape_3d`` and ``rism``, close the
#: frame -- ``rism`` last, the historical tail of the vocabulary).
FEATURE_FAMILIES: tuple[str, ...] = (
    "descriptors",
    "atompair_count",
    "ecfp",
    "ecfp3",
    "avalon_count",
    "rdkit_fingerprint",
    "morgan_full",
    "morgan_scaffold",
    "maccs",
    "fcfp",
    "fcfp_count",
    "topological",
    "topological_count",
    "rdkit_count",
    "pattern",
    "layered",
    "estate",
    "erg",
    "shape_3d",
    "rism",
)

#: Column-name prefix of every family (unique, no prefix is a prefix of
#: another, so ``family_of_column`` is an exact inverse).
FAMILY_PREFIXES: dict[str, str] = {
    "descriptors": "desc_",
    "atompair_count": "apc_bit_",
    "ecfp": "ecfp_bit_",
    "ecfp3": "ecfp3_bit_",
    "avalon_count": "avalon_count_",
    "rdkit_fingerprint": "rdkit_bit_",
    "morgan_full": "morgan_full_bit_",
    "morgan_scaffold": "morgan_scaffold_bit_",
    "maccs": "maccs_bit_",
    "fcfp": "fcfp_bit_",
    "fcfp_count": "fcfp_count_",
    "topological": "topo_bit_",
    "topological_count": "topo_count_",
    "rdkit_count": "rdkit_count_",
    "pattern": "pattern_bit_",
    "layered": "layered_bit_",
    "estate": "estate_bit_",
    "erg": "erg_",
    "shape_3d": "d3d_",
    "rism": "rism_",
}

#: Fold length of the sparse atom-pair count fingerprint.
ATOMPAIR_BITS = 2048
#: Bit length of the ECFP (Morgan, radius 2) binary fingerprint.
ECFP_BITS = 2048
#: Bit length of the radius-3 ECFP fingerprint (``Ml/D2.py``'s own R3
#: twin -- the dual-radius story the upload trained its D2 model on).
ECFP3_BITS = 2048
#: Morgan radius of the :data:`ecfp3` family.
ECFP3_RADIUS = 3
#: Bit length of the Avalon count fingerprint (the rar's 512-bit choice).
AVALON_BITS = 512
#: Bit length of the path-based RDKit fingerprint.
RDKIT_FP_BITS = 2048
#: Bit length of both Morgan bit fingerprints (full molecule and scaffold).
MORGAN_BITS = 2048
#: Morgan fingerprint radius (ECFP-style).
MORGAN_RADIUS = 2
#: Bit length of the MACCS structure-key fingerprint (RDKit's fixed 167
#: SMARTS keys).
MACCS_BITS = 167
#: Bit length of the FCFP (feature-invariant Morgan) fingerprint, both
#: the bit and the count flavour.
FCFP_BITS = 2048
#: Bit length of the topological-torsion fingerprint, both flavours
#: (datamol's 'topological' -- the 4-atom torsion generator).
TOPOLOGICAL_BITS = 2048
#: Bit length of the pattern fingerprint (substructure screening bits).
PATTERN_BITS = 2048
#: Bit length of the layered fingerprint.
LAYERED_BITS = 2048
#: Length of the E-State fingerprint: the 79 atom-type counts of
#: ``Chem.EState.Fingerprinter.FingerprintMol`` (NOT the twin E-State
#: SUMS vector).
ESTATE_BITS = 79
#: Length of the ERG (extended-connectivity reduced graph) fingerprint
#: -- 315 on this RDKit build (2026.09.1), checked per molecule.
ERG_BITS = 315
#: Number of 3D shape descriptor columns (the ``shape_3d`` family; see
#: :data:`pharmaml.chem.descriptors.D3D_COLUMNS`).
D3D_DESCRIPTOR_COUNT = 9

#: Number of RISM-KH solvation descriptor columns (see
#: :data:`pharmaml.chem.rism.RISM_DESCRIPTORS`).
RISM_DESCRIPTOR_COUNT = 10

#: Width of every family, keyed by family name.
_FAMILY_BITS: dict[str, int] = {
    "atompair_count": ATOMPAIR_BITS,
    "ecfp": ECFP_BITS,
    "ecfp3": ECFP3_BITS,
    "avalon_count": AVALON_BITS,
    "rdkit_fingerprint": RDKIT_FP_BITS,
    "morgan_full": MORGAN_BITS,
    "morgan_scaffold": MORGAN_BITS,
    "maccs": MACCS_BITS,
    "fcfp": FCFP_BITS,
    "fcfp_count": FCFP_BITS,
    "topological": TOPOLOGICAL_BITS,
    "topological_count": TOPOLOGICAL_BITS,
    "rdkit_count": RDKIT_FP_BITS,
    "pattern": PATTERN_BITS,
    "layered": LAYERED_BITS,
    "estate": ESTATE_BITS,
    "erg": ERG_BITS,
    "shape_3d": D3D_DESCRIPTOR_COUNT,
    "rism": RISM_DESCRIPTOR_COUNT,
}

#: Reference bit lengths recorded by :func:`selection_metadata` (the
#: seventeen fingerprint families; ``descriptors``, ``shape_3d`` and
#: ``rism`` are descriptor families, covered by ``families_computed``
#: instead).
_FINGERPRINT_SIZES: dict[str, int] = {
    "ecfp": ECFP_BITS,
    "ecfp3": ECFP3_BITS,
    "rdkit_fingerprint": RDKIT_FP_BITS,
    "morgan_full": MORGAN_BITS,
    "morgan_scaffold": MORGAN_BITS,
    "avalon_count": AVALON_BITS,
    "atompair_count": ATOMPAIR_BITS,
    "maccs": MACCS_BITS,
    "fcfp": FCFP_BITS,
    "fcfp_count": FCFP_BITS,
    "topological": TOPOLOGICAL_BITS,
    "topological_count": TOPOLOGICAL_BITS,
    "rdkit_count": RDKIT_FP_BITS,
    "pattern": PATTERN_BITS,
    "layered": LAYERED_BITS,
    "estate": ESTATE_BITS,
    "erg": ERG_BITS,
}

#: Column names of the ``shape_3d`` family: the nine
#: :data:`pharmaml.chem.descriptors.D3D_COLUMNS` names, snake_cased and
#: re-prefixed with ``d3d_`` so the frame vocabulary stays one flat
#: prefix namespace (``RadiusOfGyration`` -> ``d3d_radius_of_gyration``).
_D3D_FAMILY_COLUMNS: tuple[str, ...] = (
    "d3d_asphericity",
    "d3d_eccentricity",
    "d3d_pmi1",
    "d3d_pmi2",
    "d3d_pmi3",
    "d3d_radius_of_gyration",
    "d3d_spherocity_index",
    "d3d_npr1",
    "d3d_npr2",
)

_RANK_METHODS: tuple[str, ...] = ("auto", "random_forest", "lightgbm", "variance")
_RANK_TASKS: tuple[str, ...] = ("auto", "classification", "regression")


@dataclass(frozen=True, slots=True)
class FeatureReport:
    """Audit record for one :func:`compute_feature_frame` run.

    Attributes
    ----------
    smiles:
        SMILES of every molecule that produced a feature row, in row order
        (matches the frame's ``original_index`` order).
    dropped:
        ``(row, smiles)`` pairs for molecules that failed to parse or
        compute, with the ORIGINAL input row index -- align a target
        column against the frame index plus these rows.
    families:
        Mapping family name -> number of columns produced, in canonical
        :data:`FEATURE_FAMILIES` order (only the requested families
        appear).
    """

    smiles: list[str]
    dropped: list[tuple[int, str]]
    families: dict[str, int]

    @property
    def total_columns(self) -> int:
        """Total number of feature columns across all computed families."""
        return sum(self.families.values())

    @property
    def n_valid(self) -> int:
        """Number of molecules that produced a feature row."""
        return len(self.smiles)


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """Result of :func:`select_top_features`.

    Attributes
    ----------
    scores:
        The importance scores the selection was based on (as passed in).
    selected:
        Selected column names in rank order -- highest score first, ties
        broken by frame column order.
    family_counts:
        How many selected columns come from each family, in canonical
        :data:`FEATURE_FAMILIES` order (families with no selected column
        are omitted).
    method:
        Label for the ranking method that produced ``scores`` (recorded
        as given, e.g. ``"random_forest"`` or a caller-combined label).
    """

    scores: pd.Series
    selected: list[str]
    family_counts: dict[str, int]
    method: str


def family_of_column(column: str) -> str:
    """Return the feature family a frame column belongs to.

    The inverse of :data:`FAMILY_PREFIXES`: ``"desc_exactmw"`` ->
    ``"descriptors"``, ``"ecfp_bit_17"`` -> ``"ecfp"``.

    Parameters
    ----------
    column:
        Column name of a :func:`compute_feature_frame` frame.

    Returns
    -------
    str
        The family name (one of :data:`FEATURE_FAMILIES`).

    Raises
    ------
    ValueError
        If the column does not start with any known family prefix (a
        ``SMILES`` or ``Target`` column, a typo, an empty string).

    Examples
    --------
    >>> family_of_column("apc_bit_2047")
    'atompair_count'
    """
    for family in FEATURE_FAMILIES:
        if column.startswith(FAMILY_PREFIXES[family]):
            return family
    raise ValueError(
        f"unknown feature-column prefix in {column!r}; columns must start with "
        "one of: "
        + ", ".join(FAMILY_PREFIXES[family] for family in FEATURE_FAMILIES)
    )


def _as_smiles_list(smiles: Iterable[str] | pd.Series, arg_name: str) -> list[str]:
    """Materialise a SMILES sequence once; reject a bare string.

    ``compute_feature_frame("CCO")`` would otherwise iterate the string
    character-by-character into three single-atom "molecules" (the same
    trap :mod:`pharmaml.chem.descriptors` guards against).
    """
    if isinstance(smiles, str):
        preview = smiles if len(smiles) <= 40 else smiles[:40] + "..."
        # ValueError (not TypeError) by contract: the API's error currency.
        raise ValueError(  # noqa: TRY004 — the contract pins ValueError
            f"{arg_name} must be a list or pandas.Series of SMILES strings, not a "
            f"single string (got {preview!r}); wrap it in a list to compute "
            "one molecule"
        )
    return list(smiles)


def _descriptor_computer() -> tuple[list[str], Callable[[Any], np.ndarray]]:
    """Descriptor family: every RDKit available property, ``desc_``-prefixed."""
    from rdkit.Chem import rdMolDescriptors

    property_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    property_getter = rdMolDescriptors.Properties(property_names)

    def descriptor_vector(mol: Any) -> np.ndarray:
        # NaN property values (if RDKit ever emits them) are preserved --
        # filling happens only inside rank_features, by contract.
        return np.asarray(property_getter.ComputeProperties(mol), dtype=float)

    return property_names, descriptor_vector


def _atompair_computer() -> Callable[[Any], np.ndarray]:
    """Atom-pair family: sparse counts folded modulo 2048 (count semantics)."""
    from rdkit.Chem import rdFingerprintGenerator

    atompair_generator = rdFingerprintGenerator.GetAtomPairGenerator()

    def atompair_vector(mol: Any) -> np.ndarray:
        vector = np.zeros(ATOMPAIR_BITS, dtype=float)
        sparse = atompair_generator.GetSparseCountFingerprint(mol)
        for index, count in sparse.GetNonzeroElements().items():
            vector[index % ATOMPAIR_BITS] += count
        return vector

    return atompair_vector


def _avalon_computer() -> Callable[[Any], np.ndarray]:
    """Avalon family: the 512-bit Avalon count fingerprint."""
    from rdkit.Avalon import pyAvalonTools

    def avalon_vector(mol: Any) -> np.ndarray:
        vector = np.zeros(AVALON_BITS, dtype=float)
        counts = pyAvalonTools.GetAvalonCountFP(mol, nBits=AVALON_BITS)
        for index, count in counts.GetNonzeroElements().items():
            vector[index % AVALON_BITS] += count
        return vector

    return avalon_vector


def _onbits_vector(fingerprint: Any, n_bits: int) -> np.ndarray:
    """Dense float vector of an ``ExplicitBitVect`` (1.0 on set positions)."""
    vector = np.zeros(n_bits, dtype=float)
    for index in fingerprint.GetOnBits():
        vector[index] = 1.0
    return vector


def _morgan_computer() -> Callable[[Any], np.ndarray]:
    """Morgan family: radius-2 2048-bit bit vector (chem/descriptors style)."""
    from rdkit.Chem import rdMolDescriptors

    def morgan_vector(mol: Any) -> np.ndarray:
        # The legacy GetMorganFingerprintAsBitVect call is deliberate: it
        # matches pharmaml.chem.descriptors.compute_fingerprints.  Spelled
        # through rdMolDescriptors (the defining module) instead of AllChem's
        # re-export so mypy resolves the attribute the same way.
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS
        )
        return _onbits_vector(fingerprint, MORGAN_BITS)

    return morgan_vector


def _ecfp3_computer() -> Callable[[Any], np.ndarray]:
    """ECFP3 family: radius-3 2048-bit bit vector (``Ml/D2.py``'s R3).

    The upload's dual-radius fingerprints were Morgan radius 2 (4096
    bits) + radius 3 (2048 bits); this family reproduces the R3 half
    at the toolkit's own 2048-bit width, capturing the wider atom
    environments radius 2 cannot see.  Same modern
    ``rdMolDescriptors`` spelling as :func:`_morgan_computer`.
    """
    from rdkit.Chem import rdMolDescriptors

    def ecfp3_vector(mol: Any) -> np.ndarray:
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=ECFP3_RADIUS, nBits=ECFP3_BITS
        )
        return _onbits_vector(fingerprint, ECFP3_BITS)

    return ecfp3_vector


# ---------------------------------------------------------------------
# The ten fak_ml.ipynb families (cell 40's datamol fingerprint types)
# ---------------------------------------------------------------------


def _maccs_computer() -> Callable[[Any], np.ndarray]:
    """MACCS family: the fixed 167 structure-key bits."""
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    def maccs_vector(mol: Any) -> np.ndarray:
        fingerprint = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        # DataStructs.ConvertToNumpyArray fills an existing 1-D array
        # (the descriptors module's own MACCS path).
        buffer = np.zeros(MACCS_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, buffer)
        return buffer.astype(float)

    return maccs_vector


def _fcfp_generator() -> Any:
    """The feature-invariant Morgan generator (radius 2, 2048 bits).

    ``GetMorganFeatureAtomInvGen`` swaps the connectivity invariants
    for Gasteiger-Marsili feature labels (donor/acceptor/aromatic/...)
    -- the FCFP/ECFC distinction in the fak_ml notebook's own spelling.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=FCFP_BITS,
        includeChirality=False,
        useBondTypes=True,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )


def _fcfp_computer(generator: Any) -> Callable[[Any], np.ndarray]:
    """FCFP family: feature-invariant Morgan bits (2048)."""

    def fcfp_vector(mol: Any) -> np.ndarray:
        return _onbits_vector(generator.GetFingerprint(mol), FCFP_BITS)

    return fcfp_vector


def _fcfp_count_computer(generator: Any) -> Callable[[Any], np.ndarray]:
    """FCFP count family: folded feature-invariant Morgan counts (2048).

    ``GetCountFingerprintAsNumPy`` is this build's spelling of the
    folded count vector (the generator object carries no
    ``GetCountFingerprintAsBitVect`` on RDKit 2026.09).
    """

    def fcfp_count_vector(mol: Any) -> np.ndarray:
        return np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=float)

    return fcfp_count_vector


def _torsion_generator() -> Any:
    """The topological-torsion generator (datamol's 'topological').

    The fak_ml notebook benchmarks ``dm.to_fp(fp_type='topological')``;
    datamol's own ``fp.py`` maps that name to
    ``rdFingerprintGenerator.GetTopologicalTorsionGenerator`` -- the
    4-atom torsion fingerprint with ``countSimulation=True`` -- NOT the
    RDKit path fingerprint (a first draft here made that mistake; the
    torsion generator reproduces datamol's bits byte-exactly, verified
    on ethanol and aspirin).  Both the ``topological`` and
    ``topological_count`` families ride this generator with datamol's
    exact default arguments.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=False,
        torsionAtomCount=4,
        countSimulation=True,
        countBounds=None,
        fpSize=TOPOLOGICAL_BITS,
        atomInvariantsGenerator=None,
    )


def _rdkit_count_generator() -> Any:
    """The RDKit path generator at ONE bit per feature (datamol's count).

    datamol's ``rdkit-count`` default is ``numBitsPerFeature=1`` (the
    bit flavour 'rdkit' uses 2) -- the count vector therefore counts
    each PATH once instead of double-counting its two set bits, which
    is why a naive ``GetRDKitFPGenerator(fpSize=2048)`` count comes out
    at exactly twice datamol's (aspirin: 602 vs 301 -- probed).  This
    generator reproduces the notebook's numbers exactly.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetRDKitFPGenerator(
        fpSize=RDKIT_FP_BITS, numBitsPerFeature=1
    )


def _generator_bit_computer(generator: Any, bits: int) -> Callable[[Any], np.ndarray]:
    """Bit-vector computer over an ``rdFingerprintGenerator`` generator."""

    def bit_vector(mol: Any) -> np.ndarray:
        return _onbits_vector(generator.GetFingerprint(mol), bits)

    return bit_vector


def _generator_count_computer(generator: Any) -> Callable[[Any], np.ndarray]:
    """Count-vector computer over an ``rdFingerprintGenerator`` generator.

    The count flavour shares the honest build note of
    :func:`_fcfp_count_computer` (``GetCountFingerprintAsNumPy``).
    """

    def count_vector(mol: Any) -> np.ndarray:
        return np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=float)

    return count_vector


def _pattern_computer() -> Callable[[Any], np.ndarray]:
    """Pattern family: the substructure-screening pattern bits (2048)."""
    from rdkit import Chem

    def pattern_vector(mol: Any) -> np.ndarray:
        fingerprint = Chem.PatternFingerprint(mol, fpSize=PATTERN_BITS)
        return _onbits_vector(fingerprint, PATTERN_BITS)

    return pattern_vector


def _layered_computer() -> Callable[[Any], np.ndarray]:
    """Layered family: the layered (multi-property path) bits (2048)."""
    from rdkit import Chem

    def layered_vector(mol: Any) -> np.ndarray:
        fingerprint = Chem.LayeredFingerprint(mol, fpSize=LAYERED_BITS)
        return _onbits_vector(fingerprint, LAYERED_BITS)

    return layered_vector


def _estate_computer() -> Callable[[Any], np.ndarray]:
    """E-State family: the 79 atom-type COUNTS as float.

    ``Chem.EState.Fingerprinter.FingerprintMol`` returns ``(counts,
    sums)``; this family keeps the COUNTS vector (the honest choice to
    document: the E-State fingerprint IS 79 atom-type counts) and leaves
    the twin E-State SUMS vector unshipped.  Entries are small
    non-negative integers -- 0/1 for most molecules, higher when an
    atom environment repeats -- cast to float like every other family;
    the ``estate_bit_`` prefix follows the notebook's own family name.
    """
    from rdkit.Chem.EState import Fingerprinter

    def estate_vector(mol: Any) -> np.ndarray:
        counts, _sums = Fingerprinter.FingerprintMol(mol)
        return np.asarray(counts, dtype=float)

    return estate_vector


def _erg_computer() -> Callable[[Any], np.ndarray]:
    """ERG family: the reduced-graph fingerprint (315 on this build).

    ``rdReducedGraphs.GetErGFingerprint`` returns a float vector of
    extended-connectivity reduced-graph contributions.  Exotic
    molecules can yield +/-inf; those cells become NaN (preserved per
    the frame contract -- :func:`rank_features` fills with 0) instead
    of poisoning the whole column.
    """
    from rdkit.Chem import rdReducedGraphs

    def erg_vector(mol: Any) -> np.ndarray:
        fingerprint = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=float)
        if fingerprint.shape != (ERG_BITS,):
            raise ValueError(
                f"ERG fingerprint has length {fingerprint.shape[0]}, expected {ERG_BITS}"
            )
        return np.where(np.isfinite(fingerprint), fingerprint, np.nan)

    return erg_vector


def _shape3d_computer(n_conformers: int = 3) -> Callable[[Any], np.ndarray]:
    """3D shape family: the nine ``d3d_*`` descriptors per molecule.

    The ``shape_3d`` family (``xxx.ipynb`` cell 6's 3D block:
    ETKDGv3 conformers at the fixed seed 42, MMFF94s->UFF fallback, 9
    rdMolDescriptors shape descriptors) rides
    :func:`pharmaml.chem.descriptors._shape_vector`.  A molecule whose
    3D pipeline fails keeps its ROW as an all-NaN vector -- the
    notebook's failed-molecule behaviour -- instead of a drop.
    """
    from pharmaml.chem.descriptors import _shape_computer

    return _shape_computer(n_conformers)


def _morgan_scaffold_computer(
    morgan_vector: Callable[[Any], np.ndarray],
) -> Callable[[Any], np.ndarray]:
    """Scaffold family: Morgan bits of the Murcko scaffold (zero row if empty)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaffold_vector(mol: Any) -> np.ndarray:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            # Acyclic molecule: the empty scaffold is NOT a failure -- the
            # contract keeps a zero row instead of dropping the molecule.
            return np.zeros(MORGAN_BITS, dtype=float)
        return morgan_vector(scaffold)

    return scaffold_vector


def compute_feature_frame(
    smiles: Sequence[str],
    families: Iterable[str] = FEATURE_FAMILIES,
    n_conformers: int = 3,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Compute a multi-family feature matrix for a batch of SMILES.

    Parameters
    ----------
    smiles:
        SMILES strings, one per molecule (a DataFrame column works
        directly).  A bare string is rejected -- wrap it in a list.
    families:
        Iterable of family names (default: all of
        :data:`FEATURE_FAMILIES`).  Unknown names raise; duplicates are
        ignored.  The output column order is the CANONICAL
        :data:`FEATURE_FAMILIES` order regardless of the request order.
    n_conformers:
        ETKDG conformer count for the ``shape_3d`` family only (the
        ``xxx.ipynb`` cell 6 default 3); every other family ignores it.

    Returns
    -------
    tuple
        ``(frame, report)``.  ``frame`` is a float :class:`pandas.DataFrame`
        with one row per molecule that parsed AND computed cleanly --
        indexed by the original input row position under the index name
        ``"original_index"`` -- and one column per requested feature,
        named with the family prefix (``desc_amw``, ``ecfp_bit_17``,
        ``avalon_count_511``, ...).  NaN values are preserved (no filling
        here; :func:`rank_features` fills with 0).  ``report`` is a
        :class:`FeatureReport` with the kept SMILES, the dropped
        ``(row, smiles)`` pairs and the per-family column counts.  When
        every molecule fails, a 0-row frame (with all columns) plus the
        fully-populated report is returned -- check ``report.n_valid``.
        The ``shape_3d`` family is the one deliberate exception to the
        drop rule: a molecule whose 3D pipeline fails keeps its row with
        NaN ``d3d_*`` cells (the notebook's failed-molecule behaviour).

    Raises
    ------
    ValueError
        If ``smiles`` is a bare string or empty, ``families`` is a bare
        string, ``None`` or empty, or contains a name outside
        :data:`FEATURE_FAMILIES`.
    DependencyError
        If RDKit is not installed (raised by the shared
        :func:`pharmaml.chem.descriptors.require_rdkit` guard).

    Notes
    -----
    Per-family computation (the RDKit 2026.09 spellings):

    * ``descriptors``: ``rdMolDescriptors.Properties`` over
      ``GetAvailableProperties()`` (43 columns on this build), named
      ``desc_<property>``.
    * ``atompair_count``:
      ``rdFingerprintGenerator.GetAtomPairGenerator().GetSparseCountFingerprint``
      folded by ``index % 2048`` with count semantics (collisions and
      repeated environments SUM; the deprecated ``GetAtomPairFingerprint``
      and the rar's datamol shortcut are both avoided).
    * ``ecfp`` / ``morgan_full``: ``GetMorganFingerprintAsBitVect`` at
      radius 2, 2048 bits.  ``ecfp3``: the same call at radius 3
      (2048 bits) -- ``Ml/D2.py``'s wider-environment twin.
    * ``avalon_count``: ``pyAvalonTools.GetAvalonCountFP(mol, nBits=512)``.
    * ``rdkit_fingerprint``: ``Chem.RDKFingerprint(mol, fpSize=2048)``.
    * ``morgan_scaffold``: Murcko scaffold via
      ``MurckoScaffold.GetScaffoldForMol`` (NOT the rar's
      ``GetMurckoScaffold``, which no longer exists), then Morgan bits;
      an empty (acyclic) scaffold yields a zero row, never a drop.
    * ``maccs``: ``rdMolDescriptors.GetMACCSKeysFingerprint`` -> the
      167 structure-key bits (``maccs_bit_*``).
    * ``fcfp`` / ``fcfp_count``: the feature-invariant Morgan
      generator (``GetMorganGenerator(radius=2, fpSize=2048,
      atomInvariantsGenerator=GetMorganFeatureAtomInvGen())``), bit
      flavour via ``GetFingerprint`` and count flavour via
      ``GetCountFingerprintAsNumPy``.
    * ``topological`` / ``topological_count``: the topological-TORSION
      generator ``GetTopologicalTorsionGenerator(torsionAtomCount=4,
      countSimulation=True, fpSize=2048)`` -- datamol's own mapping for
      its 'topological' fp type (the notebook's benchmark vocabulary),
      verified byte-exact against ``dm.to_fp`` on ethanol and aspirin;
      bit flavour via ``GetFingerprint``, count flavour via
      ``GetCountFingerprintAsNumPy``.
    * ``rdkit_count``: the RDKit path generator at
      ``numBitsPerFeature=1`` (datamol's 'rdkit-count' default -- the
      bit flavour sets TWO bits per path, so a default-generator count
      double-counts; this spelling counts each path once and matches
      ``dm.to_fp(fp_type='rdkit-count')`` exactly).
    * ``pattern``: ``Chem.PatternFingerprint(mol, fpSize=2048)``;
      ``layered``: ``Chem.LayeredFingerprint(mol, fpSize=2048)``.
    * ``estate``: ``Chem.EState.Fingerprinter.FingerprintMol`` -> the
      79 atom-type COUNTS (not the twin SUMS vector) as float.
    * ``erg``: ``rdReducedGraphs.GetErGFingerprint`` -> 315 floats on
      this build (length checked per molecule; non-finite cells for
      exotic molecules become NaN, preserved per the frame contract).
    * ``shape_3d``: the nine 3D shape descriptors of
      :func:`pharmaml.chem.descriptors.compute_3d_descriptors`
      (``xxx.ipynb`` cell 6's 3D block: ETKDGv3 conformers at the fixed
      seed 42, MMFF94s->UFF fallback), named ``d3d_asphericity`` ...
      ``d3d_npr2``; a failed 3D pipeline keeps a NaN row.
    * ``rism``: the ten 3D-RISM-KH-inspired solvation descriptors from
      :mod:`pharmaml.chem.rism` (embedded 3D geometry + Gasteiger
      charges, pairwise tempered-KH surrogate), named
      ``rism_gsolv_kcalmol`` ... ``rism_q_neg_frac``.
    """
    smiles_list = _as_smiles_list(smiles, "smiles")
    # Cheap validation BEFORE any parsing: caller bugs must not wait for a
    # full RDKit pass to surface (the chem/descriptors.py ordering lesson).
    if not smiles_list:
        raise ValueError("smiles is empty -- pass at least one SMILES string")
    if families is None or isinstance(families, str):
        raise ValueError(
            "families must be an iterable of family names (e.g. "
            "families=('descriptors', 'ecfp')), not a single string or None; "
            f"got {families!r}"
        )
    family_list = list(families)
    if not family_list:
        raise ValueError("families is empty -- pass at least one family name")
    unknown = sorted(set(family_list) - set(FEATURE_FAMILIES))
    if unknown:
        raise ValueError(
            f"unknown feature family {unknown[0]!r}; valid families are: "
            + ", ".join(FEATURE_FAMILIES)
        )
    if (
        not isinstance(n_conformers, (int, np.integer))
        or isinstance(n_conformers, bool)
        or int(n_conformers) < 1
    ):
        raise ValueError(f"n_conformers must be an integer >= 1, got {n_conformers!r}")
    requested_set = set(family_list)
    requested = [family for family in FEATURE_FAMILIES if family in requested_set]

    Chem = require_rdkit()
    computers, columns, family_widths = _family_plan(requested, int(n_conformers))

    rows: list[np.ndarray] = []
    valid_smiles: list[str] = []
    valid_positions: list[int] = []
    dropped: list[tuple[int, str]] = []
    for row, smi in enumerate(smiles_list):
        text = smi if isinstance(smi, str) else str(smi)
        try:
            # Blank cells parse as valid 0-atom molecules in RDKit; they are
            # failures here (the chem/descriptors.py convention).
            mol = (
                Chem.MolFromSmiles(text)
                if isinstance(smi, str) and smi.strip()
                else None
            )
            if mol is None:
                raise ValueError("SMILES did not parse to a valid molecule")
            rows.append(np.concatenate([computers[family](mol) for family in requested]))
            valid_smiles.append(text)
            valid_positions.append(row)
        except Exception as exc:  # noqa: BLE001 — RDKit raises a zoo per molecule
            _LOG.warning(
                "Dropping molecule at row %d (%r): feature computation failed "
                "(%s: %s)",
                row,
                text,
                type(exc).__name__,
                exc,
            )
            dropped.append((row, text))

    if rows:
        matrix = np.vstack(rows)
    else:
        matrix = np.zeros((0, len(columns)), dtype=float)
    frame = pd.DataFrame(matrix, columns=columns, dtype=float)
    frame.index = pd.Index(valid_positions, name="original_index")
    report = FeatureReport(smiles=valid_smiles, dropped=dropped, families=family_widths)
    _LOG.info(
        "Computed %d feature columns for %d molecules (%d dropped)",
        frame.shape[1],
        len(valid_smiles),
        len(dropped),
    )
    return frame, report


def _family_plan(
    requested: list[str],
    n_conformers: int = 3,
) -> tuple[dict[str, Callable[[Any], np.ndarray]], list[str], dict[str, int]]:
    """Build per-family computers, column names and widths for the request."""
    computers: dict[str, Callable[[Any], np.ndarray]] = {}
    property_names: list[str] = []
    if "descriptors" in requested:
        property_names, computers["descriptors"] = _descriptor_computer()
    if "atompair_count" in requested:
        computers["atompair_count"] = _atompair_computer()
    if "avalon_count" in requested:
        computers["avalon_count"] = _avalon_computer()
    if "ecfp" in requested or "morgan_full" in requested or "morgan_scaffold" in requested:
        morgan_vector = _morgan_computer()
        if "ecfp" in requested:
            computers["ecfp"] = morgan_vector
        if "morgan_full" in requested:
            computers["morgan_full"] = morgan_vector
        if "morgan_scaffold" in requested:
            computers["morgan_scaffold"] = _morgan_scaffold_computer(morgan_vector)
    if "ecfp3" in requested:
        computers["ecfp3"] = _ecfp3_computer()
    if "maccs" in requested:
        computers["maccs"] = _maccs_computer()
    if "fcfp" in requested or "fcfp_count" in requested:
        fcfp_generator = _fcfp_generator()
        if "fcfp" in requested:
            computers["fcfp"] = _fcfp_computer(fcfp_generator)
        if "fcfp_count" in requested:
            computers["fcfp_count"] = _fcfp_count_computer(fcfp_generator)
    if "topological" in requested or "topological_count" in requested:
        torsion_generator = _torsion_generator()
        if "topological" in requested:
            computers["topological"] = _generator_bit_computer(
                torsion_generator, TOPOLOGICAL_BITS
            )
        if "topological_count" in requested:
            computers["topological_count"] = _generator_count_computer(
                torsion_generator
            )
    if "rdkit_count" in requested:
        computers["rdkit_count"] = _generator_count_computer(
            _rdkit_count_generator()
        )
    if "pattern" in requested:
        computers["pattern"] = _pattern_computer()
    if "layered" in requested:
        computers["layered"] = _layered_computer()
    if "estate" in requested:
        computers["estate"] = _estate_computer()
    if "erg" in requested:
        computers["erg"] = _erg_computer()
    if "shape_3d" in requested:
        computers["shape_3d"] = _shape3d_computer(n_conformers)

    if "rdkit_fingerprint" in requested:
        from rdkit import Chem

        def rdkit_vector(mol: Any) -> np.ndarray:
            bits = Chem.RDKFingerprint(mol, fpSize=RDKIT_FP_BITS)
            return _onbits_vector(bits, RDKIT_FP_BITS)

        computers["rdkit_fingerprint"] = rdkit_vector

    if "rism" in requested:
        # 3D-RISM-KH solvation descriptors: the pairwise surrogate from
        # pharmaml.chem.rism (embedding + Gasteiger charges inside).
        from pharmaml.chem.rism import _rism_computer

        computers["rism"] = _rism_computer()

    columns: list[str] = []
    family_widths: dict[str, int] = {}
    for family in requested:
        if family == "descriptors":
            width = len(property_names)
            columns.extend(f"desc_{name}" for name in property_names)
        elif family == "rism":
            # Import stays local so a missing/unimportable rism module
            # surfaces only when the family is requested.
            from pharmaml.chem.rism import _rism_family_columns

            names = _rism_family_columns()
            width = len(names)
            columns.extend(names)
        elif family == "shape_3d":
            names = list(_D3D_FAMILY_COLUMNS)
            width = len(names)
            columns.extend(names)
        else:
            width = _FAMILY_BITS[family]
            prefix = FAMILY_PREFIXES[family]
            columns.extend(f"{prefix}{index}" for index in range(width))
        family_widths[family] = width
    return computers, columns, family_widths


def _lightgbm_estimators() -> tuple[Any, Any] | None:
    """Return ``(LGBMClassifier, LGBMRegressor)`` or None when lightgbm is absent.

    Imported lazily on every call: lightgbm is an OPTIONAL dependency and
    must never appear on this module's import path (a ``None`` in
    ``sys.modules`` from a test monkeypatch also lands here).
    """
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        return None
    return LGBMClassifier, LGBMRegressor


def _random_forest_model(task: str, random_state: int) -> Any:
    """Instantiate the random-forest estimator for ``task`` (lazy sklearn)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    estimator_class = (
        RandomForestClassifier if task == "classification" else RandomForestRegressor
    )
    return estimator_class(n_estimators=200, random_state=random_state)


def _lightgbm_model(task: str, random_state: int) -> Any | None:
    """Instantiate the LightGBM estimator for ``task`` or None if missing."""
    estimators = _lightgbm_estimators()
    if estimators is None:
        return None
    classifier, regressor = estimators
    estimator_class = classifier if task == "classification" else regressor
    return estimator_class(n_estimators=200, random_state=random_state, verbosity=-1)


def _validated_target(
    frame: pd.DataFrame, target: Sequence[float] | None
) -> pd.Series | None:
    """Materialise ``target`` and guard length/NaN/inf BEFORE any model fit."""
    if target is None:
        return None
    y_series = pd.Series(list(target))
    if len(y_series) != len(frame):
        raise ValueError(
            f"target length {len(y_series)} does not match frame length {len(frame)}"
        )
    if bool(y_series.isna().any()):
        raise ValueError(
            "target contains NaN/None values -- clean or drop them before ranking"
        )
    numeric = pd.to_numeric(y_series, errors="coerce")
    if bool(np.isinf(numeric.to_numpy(dtype=float)).any()):
        raise ValueError("target contains non-finite values (+/-inf)")
    return y_series


def _infer_task(y_series: pd.Series) -> str:
    """classification iff the target is non-numeric or has <= 2 unique values."""
    numeric = pd.to_numeric(y_series, errors="coerce")
    non_numeric = bool((numeric.isna() & y_series.notna()).any())
    if non_numeric or int(y_series.nunique()) <= 2:
        return "classification"
    return "regression"


def _model_importances(
    frame: pd.DataFrame,
    y_series: pd.Series,
    *,
    method: str,
    task: str,
    random_state: int,
) -> pd.Series:
    """Fit the chosen estimator and return its per-column importances."""
    x_matrix = frame.fillna(0).astype(float).to_numpy()
    y_values = y_series.to_numpy()
    model: Any
    if method == "lightgbm":
        model = _lightgbm_model(task, random_state)
        if model is None:
            _LOG.info(
                "lightgbm is not installed -- falling back to random_forest ranking"
            )
            model = _random_forest_model(task, random_state)
    else:
        model = _random_forest_model(task, random_state)
    model.fit(x_matrix, y_values)
    importances = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(importances, index=frame.columns)


def _descending(scores: pd.Series) -> pd.Series:
    """Order a score series descending; ties keep their original position."""
    order = sorted(
        range(len(scores)),
        key=lambda position: (-float(scores.iloc[position]), position),
    )
    return scores.iloc[order]


def rank_features(
    frame: pd.DataFrame,
    target: Sequence[float] | None = None,
    *,
    method: str = "auto",
    task: str = "auto",
    random_state: int = 42,
) -> pd.Series:
    """Rank feature columns by importance, descending.

    Parameters
    ----------
    frame:
        Feature matrix (e.g. from :func:`compute_feature_frame`, but any
        numeric frame works).  NaNs are filled with 0 before ranking.
    target:
        Optional target values, one per frame row.  String labels (a
        ``Sequence[str]`` at runtime) select the classification task.
        ``None`` selects the variance method.
    method:
        ``'auto'`` (variance when ``target`` is None, lightgbm when
        importable, random forest otherwise), ``'variance'``,
        ``'random_forest'`` or ``'lightgbm'``.
    task:
        ``'auto'`` (classification iff the target is non-numeric or has
        <= 2 unique values), ``'classification'`` or ``'regression'``.
        Only used by the model-based methods.
    random_state:
        Seed for the model-based methods (deterministic rankings).

    Returns
    -------
    pandas.Series
        One importance per column of ``frame``, sorted descending, index
        = ``frame.columns`` (ties keep frame column order).  Variance
        scores are ``X.var(axis=0)`` over ``frame.fillna(0)`` (constant
        columns score exactly 0.0); model scores are the fitted
        estimator's ``feature_importances_``.

    Raises
    ------
    ValueError
        If ``method``/``task`` is unknown; if ``target`` is given but its
        length differs from ``frame``; or if ``target`` contains NaN,
        None or +/-inf (guarded BEFORE any model is fitted).  A
        model-based ``method`` with ``target=None`` also raises.

    Notes
    -----
    LightGBM is strictly optional: ``method='lightgbm'`` (explicitly or
    via ``'auto'``) falls back to ``'random_forest'`` -- with an info log
    line -- whenever ``import lightgbm`` fails, so the function never
    raises over a missing optional dependency.  scikit-learn is imported
    inside this call chain, keeping the module import path at
    rdkit + numpy + pandas.
    """
    if method not in _RANK_METHODS:
        raise ValueError(
            f"unknown ranking method {method!r}; valid methods: "
            + ", ".join(_RANK_METHODS)
        )
    if task not in _RANK_TASKS:
        raise ValueError(
            f"unknown task {task!r}; valid tasks: " + ", ".join(_RANK_TASKS)
        )

    y_series = _validated_target(frame, target)

    if method == "auto":
        if target is None:
            method = "variance"
        elif _lightgbm_estimators() is not None:
            method = "lightgbm"
        else:
            method = "random_forest"

    if method == "variance":
        scores = frame.fillna(0).astype(float).var(axis=0)
    else:
        if y_series is None:
            raise ValueError(
                f"method={method!r} requires a target -- pass target=... or use "
                "method='variance'"
            )
        resolved_task = task if task != "auto" else _infer_task(y_series)
        scores = _model_importances(
            frame, y_series, method=method, task=resolved_task, random_state=random_state
        )
    return _descending(scores)


def select_top_features(
    frame: pd.DataFrame,
    scores: pd.Series,
    k: int,
    *,
    median_prefilter: bool = True,
    method: str = "",
) -> FeatureSelection:
    """Select the top-K features with the rar's two-layer strategy.

    Parameters
    ----------
    frame:
        Feature matrix whose columns carry family prefixes (columns from
        :func:`compute_feature_frame`; every column must map through
        :func:`family_of_column`).
    scores:
        Importance scores, e.g. the output of :func:`rank_features`.  Its
        index must equal ``frame.columns`` exactly.
    k:
        Number of features to keep.  Must be a positive integer; ``k``
        greater than or equal to the candidate count returns ALL
        candidates in score order.
    median_prefilter:
        Layer 1 (on by default): within EACH family present in the frame,
        keep only the columns scoring >= that family's median -- the
        rar's per-family ``SelectFromModel(threshold='median')``.  With
        ``False``, every column is a candidate for layer 2.
    method:
        Label recorded verbatim on the returned :class:`FeatureSelection`
        (e.g. the ranking method name); not used for computation.

    Returns
    -------
    FeatureSelection
        The scores as passed in, the selected columns in rank order
        (layer-2 top-K with stable tie-break on frame column order), the
        per-family counts of the selected columns (canonical family
        order), and the ``method`` label.

    Raises
    ------
    ValueError
        If ``scores.index`` differs from ``frame.columns`` (set OR
        order), ``k`` is not an integer (bools rejected) or is <= 0, or a
        column does not carry a known family prefix.

    Notes
    -----
    Both layers are score-order driven and deterministic: given the same
    ``scores`` and column order, the same features are selected.  A
    single-column family always survives layer 1 (its median is its own
    score).
    """
    columns = list(frame.columns)
    score_index = list(scores.index)
    if score_index != columns:
        for position, (label, column) in enumerate(zip(score_index, columns)):
            if label != column:
                raise ValueError(
                    "scores.index must equal frame.columns exactly; mismatch at "
                    f"position {position}: {label!r} != {column!r}"
                )
        raise ValueError(
            "scores.index must equal frame.columns exactly; got "
            f"{len(score_index)} scores for {len(columns)} columns"
        )
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        # ValueError (not TypeError) by contract: k edges must raise ValueError.
        raise ValueError(f"k must be an integer, got {k!r}")  # noqa: TRY004
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    column_families = [family_of_column(column) for column in columns]
    position_of = {column: position for position, column in enumerate(columns)}

    candidates = columns
    if median_prefilter:
        family_positions: dict[str, list[int]] = {}
        for position, family in enumerate(column_families):
            family_positions.setdefault(family, []).append(position)
        kept: list[int] = []
        for positions in family_positions.values():
            median = float(np.median([float(scores.iloc[p]) for p in positions]))
            kept.extend(p for p in positions if float(scores.iloc[p]) >= median)
        candidates = [columns[position] for position in sorted(kept)]

    ordered = sorted(
        candidates,
        key=lambda column: (-float(scores.iloc[position_of[column]]), position_of[column]),
    )
    selected = ordered[: min(k, len(ordered))]

    tallies: dict[str, int] = {}
    for family in (family_of_column(column) for column in selected):
        tallies[family] = tallies.get(family, 0) + 1
    family_counts = {
        family: tallies[family] for family in FEATURE_FAMILIES if family in tallies
    }
    return FeatureSelection(
        scores=scores.copy(),
        selected=selected,
        family_counts=family_counts,
        method=method,
    )


def selection_metadata(
    selection: FeatureSelection,
    report: FeatureReport,
    *,
    k: int,
    families: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the JSON-ready metadata dict for a feature-selection run.

    Parameters
    ----------
    selection:
        The :class:`FeatureSelection` produced by
        :func:`select_top_features`.
    report:
        The :class:`FeatureReport` produced alongside the feature frame.
    k:
        The K that was requested (recorded verbatim as ``k_requested``).
    families:
        Optional iterable restricting the ``fingerprint_sizes`` entry to
        the named families (e.g. the families a GUI user ticked).  When
        ``None``, the families recorded in ``report.families`` are used.
        ``families_computed`` always mirrors ``report.families``.

    Returns
    -------
    dict
        Keys: ``selected_features`` (rank order), ``descriptor_names``
        (selected ``desc_`` columns with the prefix stripped),
        ``fingerprint_sizes`` (family -> bit length for the seventeen
        fingerprint families in scope), ``family_counts``,
        ``families_computed``, ``k_requested``, ``method``,
        ``n_molecules``, ``dropped`` (``[row, smiles]`` lists) and
        ``total_features``.  Every value is a plain Python type, so
        ``json.dumps`` works without a ``default=`` hook.

    Raises
    ------
    ValueError
        If ``families`` is a bare string (a common call-site typo that
        would otherwise silently filter by single letters).
    """
    if isinstance(families, str):
        # ValueError (not TypeError): consistent with compute_feature_frame.
        raise ValueError(  # noqa: TRY004 — consistent module error currency
            "families must be an iterable of family names, not a single string; "
            f"got {families!r}"
        )
    computed = set(families) if families is not None else set(report.families)
    descriptor_prefix = FAMILY_PREFIXES["descriptors"]
    return {
        "selected_features": list(selection.selected),
        "descriptor_names": [
            column[len(descriptor_prefix):]
            for column in selection.selected
            if column.startswith(descriptor_prefix)
        ],
        "fingerprint_sizes": {
            family: bits
            for family, bits in _FINGERPRINT_SIZES.items()
            if family in computed
        },
        "family_counts": dict(selection.family_counts),
        "families_computed": dict(report.families),
        "k_requested": int(k),
        "method": selection.method,
        "n_molecules": int(report.n_valid),
        "dropped": [[int(row), str(smi)] for row, smi in report.dropped],
        "total_features": int(report.total_columns),
    }


@dataclass(frozen=True, slots=True)
class VarianceFilterResult:
    """Result of :func:`filter_low_variance`.

    Attributes
    ----------
    frame:
        The filtered frame — kept columns only, original column order
        and original rows untouched (a view-by-copy of the input's
        surviving columns).
    kept:
        Kept column names, in frame order.
    dropped:
        Cut column name -> the variance that got it cut (every entry
        is strictly below ``threshold``, or NaN for an inf-poisoned
        column whose variance is undefined; the record carries its own
        evidence, so an audit never has to recompute).
    threshold:
        The threshold the filter ran with.
    """

    frame: pd.DataFrame
    kept: list[str]
    dropped: dict[str, float]
    threshold: float

    @property
    def n_kept(self) -> int:
        """Number of columns that survived the filter."""
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        """Number of columns the filter cut."""
        return len(self.dropped)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready view (the frame itself is not embedded)."""
        return {
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "kept": list(self.kept),
            "dropped": dict(self.dropped),
            "threshold": float(self.threshold),
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"variance filter (threshold {self.threshold:g}): kept "
            f"{self.n_kept}, dropped {self.n_dropped}"
        )


def filter_low_variance(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.01,
) -> VarianceFilterResult:
    """Cut feature columns whose variance is below ``threshold``.

    The variance-threshold step of the ``xxx.ipynb`` filtering cell
    (``VarianceThreshold(threshold=0.01)`` there, ``D2.py``'s audit GAP
    6 here): a column that never moves cannot separate actives from
    inactives, and a 27k-column frame ships hundreds of them.  The
    companion of :func:`rank_features` — that function RANKS by
    variance, this one CUTS by it, with the same variance definition
    (``frame.fillna(0).var(axis=0)``) so a ranker's scores and this
    filter's verdicts always agree.

    Parameters
    ----------
    frame:
        Feature matrix (e.g. from :func:`compute_feature_frame`).
        Non-numeric columns are NOT filtered — they pass through
        untouched with a warning (a ``SMILES``/target column riding
        along survives, the ``clean_numeric_frame`` precedent).
    threshold:
        Inclusive-exclusive cut: a column is dropped when its variance
        is STRICTLY below ``threshold`` (a variance exactly equal to
        the threshold survives, mirroring sklearn's
        ``VarianceThreshold`` semantics).  Default ``0.01``, the
        notebook's value.  Must be a real number, finite and >= 0
        (bools and numeric strings are rejected at the door -- the
        house guard family; ``float(True)`` is ``1.0`` and
        ``float('0.01')`` is ``0.01``, but neither is an honest
        threshold, 100x-153).

    Returns
    -------
    VarianceFilterResult
        The kept-column frame plus the dropped-column ledger.

    Raises
    ------
    ValueError
        If ``frame`` is not a DataFrame, has fewer than two rows
        (variance is undefined for one), or ``threshold`` is not a
        real number, is negative, or is non-finite.

    Notes
    -----
    Scope honesty: the SAME notebook cell also dropped columns with
    >30% missing values and Pearson-collinear pairs (r > 0.98) — those
    are separate concerns and stay out of this function; the missing
    fraction is a :func:`~pharmaml.ml.preprocessing.clean_numeric_frame`
    -side decision and collinearity needs the pairwise matrix.  NaN
    cells are filled with 0 BEFORE the variance is computed (the
    :func:`rank_features` convention, so the two functions can never
    disagree about a column).

    Undefined variance (100x-153): after the ``fillna(0)`` convention
    a NaN variance can only arise from inf/-inf cells — a poisoned
    column every downstream estimator in this toolkit rejects with a
    typed error.  Such columns are CUT (never silently passed) and
    ledgered as NaN — the same ``undefined`` verdict
    :func:`rank_features` already reports for them, so the twin
    functions stay in agreement.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({
    ...     "moves": [1.0, 2.0, 3.0, 4.0],       # var 1.6667
    ...     "flat": [7.0, 7.0, 7.0, 7.0],        # var 0.0
    ...     "tiny": [1.0, 1.0, 1.0, 1.005],      # var ~0.000006
    ... })
    >>> result = filter_low_variance(frame)
    >>> result.kept, list(result.dropped)
    (['moves'], ['flat', 'tiny'])
    >>> round(result.dropped["flat"], 6)
    0.0
    >>> poisoned = pd.DataFrame({
    ...     "moves": [1.0, 2.0, 3.0, 4.0],
    ...     "bad": [float("inf"), 1.0, 2.0, 3.0],
    ... })
    >>> presult = filter_low_variance(poisoned)
    >>> presult.kept, list(presult.dropped)
    (['moves'], ['bad'])
    >>> import math
    >>> math.isnan(presult.dropped["bad"])  # undefined, never silently kept
    True
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(  # noqa: TRY004 — consistent module error currency
            f"frame must be a pandas.DataFrame, got {type(frame).__name__}"
        )
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        # (100x-153) the type guard must fire BEFORE float() coerces:
        # float(True) is 1.0 and float('0.01') is 0.01, and the house
        # guard family (interpret's radius/bit, viz's top_n) never lets
        # a non-number through the door.
        raise ValueError(  # noqa: TRY004 — consistent module error currency
            f"threshold must be a real number >= 0, got {threshold!r}"
        )
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            f"threshold must be a finite number >= 0, got {threshold!r}"
        )
    if len(frame) < 2:
        raise ValueError(
            f"variance needs at least 2 rows, got {len(frame)} — a "
            "single-row frame is constant by construction"
        )

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    non_numeric = [
        column for column in frame.columns if column not in numeric_columns
    ]
    if non_numeric:
        _LOG.warning(
            "filter_low_variance: %d non-numeric column(s) pass through "
            "unfiltered: %s",
            len(non_numeric),
            non_numeric[:10],
        )
    if not numeric_columns:
        # Nothing to filter -- the honest result keeps everything
        # (there is no variance to cut on); the caller sees the same
        # frame back with an empty dropped ledger.
        return VarianceFilterResult(
            frame=frame.copy(), kept=list(frame.columns), dropped={}, threshold=threshold
        )

    variances = frame.loc[:, numeric_columns].fillna(0).astype(float).var(axis=0)
    # (100x-153) an UNDEFINED variance (NaN after the fillna(0)
    # convention — finite NaN cells became 0, so this is inf/-inf
    # poison) is CUT, never silently passed: every downstream
    # estimator in the toolkit rejects non-finite inputs with a typed
    # error, and rank_features already reports the same column as
    # NaN.  The ledger entry is NaN (undefined), the warning names
    # the poison.
    undefined = [
        column
        for column in numeric_columns
        if not np.isfinite(float(variances[column]))
    ]
    undefined_set = set(undefined)
    if undefined:
        _LOG.warning(
            "filter_low_variance: %d column(s) carry inf/-inf cells "
            "(undefined variance) -- cut: %s",
            len(undefined),
            undefined[:10],
        )
    kept_numeric = [
        column
        for column in numeric_columns
        if column not in undefined_set
        and not (float(variances[column]) < threshold)
    ]
    dropped: dict[str, float] = {}
    for column in numeric_columns:
        if column in undefined_set:
            dropped[column] = float("nan")
        elif float(variances[column]) < threshold:
            dropped[column] = float(variances[column])
    # A column survives when it is non-numeric (passes through
    # untouched, the warned-about carrier columns) OR numeric with a
    # variance at/above the threshold — frame order preserved.
    kept_set = set(kept_numeric)
    ordered = [
        column
        for column in frame.columns
        if column in kept_set
        or not pd.api.types.is_numeric_dtype(frame[column])
    ]
    return VarianceFilterResult(
        frame=frame.loc[:, ordered].copy(),
        kept=ordered,
        dropped=dropped,
        threshold=threshold,
    )

"""MOSES-lite evaluation: audit a generated SMILES list the way the benchmark would.

This is the evaluation layer the generative subpackage absorbed from
AspirinCode's drug-likeness-space (https://github.com/AspirinCode/drug-likeness-space):
its ``crt/utils.py`` ships ``get_mol`` (RDKit parse, ``None`` on failure),
``canonic_smiles`` (``MolToSmiles`` of the parsed molecule) and
``check_novelty(gen_smiles, train_smiles)`` -- the novelty percentage of the
generated set against the training set -- and its ``crt/sampling.py`` closes
every generation run with the same three numbers: valid ratio, unique SMILES,
novelty.  The metrics *standard* those numbers come from is MOSES
(https://github.com/molecularsets/moses), the molecular generation
benchmarking platform.  :func:`generation_metrics` is that flow as one call:
hand it the raw generated list plus an optional reference (training) set and
it returns a :class:`GenerationMetrics` record -- the scorecard, the internal
diversity, the scaffold diversity and the property distribution shifts.

What did NOT land (the honest section)
--------------------------------------
The full MOSES scorecard needs trained reference models or external weights,
and those did **not** land: **FCD** (Frechet ChemNet Distance) needs the
ChemNet weights and fitted reference activation statistics; **SNN**
(similarity to the nearest neighbour vs the reference set), **fragment
similarity** and the **distribution learning plots** need trained reference
models / external weights as well.  Also, the CRT torch generator itself did
not land -- this module *evaluates* generated SMILES lists (the GA / RL /
MCTS optimisers' output included); it does not generate.  What is here is the
MOSES *subset* computable from RDKit + numpy alone: valid / unique / novel
(the exact crt flow), internal diversity (IntDiv1 / IntDiv2), scaffold
diversity and the property distribution shifts (MW, MolLogP, TPSA, QED).

Design notes
------------
* **Fingerprints** are Morgan radius 2, 2048 bits -- computed with the modern
  ``rdFingerprintGenerator.GetMorganGenerator`` (bit-identical to
  ``AllChem.GetMorganFingerprintAsBitVect(radius=2, nBits=2048)`` but without
  its per-call deprecation warning, the same switch
  :mod:`~pharmaml.generative.diversity` made).  The bit vectors are stacked
  with ``DataStructs.ConvertToNumpyArray`` into an ``(n, 2048)`` uint8 matrix
  (the :func:`pharmaml.chem.descriptors.compute_fingerprints` house pattern,
  chosen over ``np.packbits``) and the pairwise Tanimoto matrix is
  :func:`pharmaml.chem.similarity.tanimoto_matrix` -- the toolkit's own O(n^2)
  numpy on stacked bit arrays (a uint32 matmul), *reused* rather than
  reimplemented; :func:`~pharmaml.chem.similarity.tanimoto_from_bitarrays` is
  the scalar sibling of the same math.
* **Scaffolds** are Murcko
  (``Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol`` + ``MolToSmiles``).
  An acyclic molecule's Murcko scaffold is the empty molecule, which
  canonicalises to ``''`` -- the ``diversity.py`` scaffold buckets treat that
  ``''`` key like any other, so it counts as one scaffold here too.
* **The pairwise cap**: the O(n^2) IntDiv loops run on at most
  :data:`PAIRWISE_CAP` (2000) unique molecules -- the *first 2000
  canonical-sorted* ones, a deterministic head, never a random subsample --
  and flag it via ``pairwise_truncated`` (and a note in ``__str__``).
  Novelty, scaffold diversity and the property shifts always see *all* the
  unique valid molecules.
* **Determinism**: everything is deterministic -- no sampling, no seeds
  anywhere -- so there is no ``random_state`` parameter to thread through.
* **Property shifts** are means over the *unique valid* canonical molecules
  of each side (duplicates carry no distributional information); ``shift`` is
  ``generated - reference``.  Without a usable reference the
  ``reference_mean`` (and therefore ``shift``) is NaN -- honest, not a fake
  zero.

Examples
--------
>>> from pharmaml.generative.moses_metrics import (
...     PAIRWISE_CAP, generation_metrics)

The crt/sampling.py flow on a mixed list -- aspirin, ethanol, benzene, a
duplicate ethanol and one unparseable string:

>>> generated = [
...     "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
...     "CCO",                     # ethanol
...     "c1ccccc1",                # benzene
...     "CCO",                     # duplicate ethanol
...     "not_a_smiles",            # unparseable
... ]
>>> metrics = generation_metrics(generated)
>>> metrics.n_generated, metrics.n_valid, metrics.valid_fraction
(5, 4, 0.8)
>>> metrics.n_unique, metrics.unique_fraction
(3, 0.75)

Novelty against a reference (training) set -- crt's
``check_novelty(unique_canonical, set(train))`` arithmetic, reported as a
fraction instead of a percentage:

>>> refd = generation_metrics(generated, reference=["CCO", "CC(=O)Oc1ccccc1C(=O)O"])
>>> refd.n_novel, round(refd.novel_fraction, 6)     # only benzene is new
(1, 0.333333)
>>> bare = generation_metrics(generated)            # no reference: honest 0.0
>>> bare.novel_fraction
0.0
>>> bool(bare.property_shifts["reference_mean"].isna().all())
True

Internal diversity over the unique valid molecules -- the mean pairwise
Tanimoto *distance* (IntDiv1) and its root-mean-square sibling (IntDiv2):

>>> 0.0 <= metrics.internal_diversity_1 <= 1.0
True
>>> round(metrics.internal_diversity_1, 4), round(metrics.internal_diversity_2, 4)
(0.9213, 0.923)
>>> twins = generation_metrics(["CCO", "CCO", "CCO"])   # one molecule: no pairs
>>> twins.internal_diversity_1, twins.internal_diversity_2
(0.0, 0.0)
>>> twins.unique_fraction                 # the duplicate collapse lands here
0.3333333333333333
>>> (twins.n_scaffolds, twins.scaffold_diversity)  # 1 scaffold / 1 unique mol
(1, 1.0)

Scaffold diversity: aspirin and benzene share the benzene ring, ethanol is
acyclic (its Murcko scaffold is the empty molecule, ``''``) -- two
scaffolds over three unique molecules:

>>> trio = generation_metrics(["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"])
>>> round(trio.scaffold_diversity, 6)
0.666667
>>> ring_kids = generation_metrics(["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"])
>>> (ring_kids.n_unique, ring_kids.n_scaffolds)
(3, 1)
>>> round(ring_kids.scaffold_diversity, 6)   # three molecules, one scaffold
0.333333

Property distribution shifts (rows MW / MolLogP / TPSA / QED, means over
the unique valid molecules, ``shift = generated - reference``):

>>> shifts = refd.property_shifts
>>> shifts.index.tolist(), shifts.columns.tolist()
(['MW', 'MolLogP', 'TPSA', 'QED'], ['generated_mean', 'reference_mean', 'shift'])
>>> [round(v, 3) for v in shifts["generated_mean"]]
[101.447, 0.998, 27.943, 0.467]
>>> [round(v, 3) for v in shifts["reference_mean"]]
[113.114, 0.654, 41.915, 0.478]
>>> [round(v, 3) for v in shifts["shift"]]
[-11.667, 0.344, -13.972, -0.012]
>>> bool(bare.property_shifts["shift"].isna().all())     # no reference: honest NaN
True
>>> metrics.as_frame() is metrics.property_shifts
True

The one-line scorecard:

>>> print(metrics)
MOSES-lite: valid 80.0% | unique 75.0% | novel 0.0% | IntDiv1 0.921 | scaffold diversity 0.667
>>> print(refd)
MOSES-lite: valid 80.0% | unique 75.0% | novel 33.3% | IntDiv1 0.921 | scaffold diversity 0.667

Degenerate inputs stay honest instead of dividing by zero:

>>> empty = generation_metrics([])
>>> (empty.n_generated, empty.n_valid, empty.valid_fraction, empty.novel_fraction)
(0, 0, 0.0, 0.0)
>>> empty.internal_diversity_1, empty.scaffold_diversity
(0.0, 0.0)
>>> bool(empty.property_shifts["generated_mean"].isna().all())
True
>>> generation_metrics(["not_a_smiles", "???"]).n_valid
0

The pairwise cap in action (lowered here so it fits in a doctest): IntDiv
runs on the first ``PAIRWISE_CAP`` canonical-sorted unique molecules:

>>> import pharmaml.generative.moses_metrics as mm
>>> saved = mm.PAIRWISE_CAP
>>> try:
...     mm.PAIRWISE_CAP = 2
...     capped = mm.generation_metrics(["CCO", "CCN", "CCS", "CCF"])
... finally:
...     mm.PAIRWISE_CAP = saved
>>> capped.n_unique, capped.pairwise_truncated
(4, True)
>>> round(capped.internal_diversity_1, 6)    # CCF + CCN, the sorted head
0.666667
>>> "capped" in str(capped)
True

A bare string is rejected (it would otherwise be iterated
character-by-character into single-atom "molecules" -- the house
``_as_smiles_list`` contract):

>>> generation_metrics("CCO")
Traceback (most recent call last):
    ...
pharmaml.exceptions.DataValidationError: generated must be a list or pandas.Series of SMILES strings, not a single string (got 'CCO'); wrap it in a list to compute one molecule
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.similarity import tanimoto_matrix
from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import canonical_smiles, require_rdkit

_LOG = get_logger(__name__)

__all__ = [
    "PAIRWISE_CAP",
    "GenerationMetrics",
    "generation_metrics",
]

#: The pairwise loops (IntDiv1 / IntDiv2) run on at most this many unique
#: molecules -- the first 2000 in canonical (sorted) order, a deterministic
#: head rather than a random subsample.  2000^2 / 2 = 2M Tanimoto pairs is
#: the point where the O(n^2) numpy matmul starts to dominate a run;
#: ``GenerationMetrics.pairwise_truncated`` flags every capped audit.
PAIRWISE_CAP: int = 2000

#: The MOSES-style property rows of ``GenerationMetrics.property_shifts``
#: (molecular weight, Crippen LogP, polar surface area, QED drug-likeness --
#: the four MOSES distribution-learning properties computable from RDKit).
_PROPERTY_ROWS: tuple[str, ...] = ("MW", "MolLogP", "TPSA", "QED")

#: Morgan fingerprint parameters (the MOSES / toolkit convention).
_MORGAN_RADIUS: int = 2
_MORGAN_BITS: int = 2048


@dataclass
class GenerationMetrics:
    """The MOSES-lite audit of one generated SMILES list.

    Attributes
    ----------
    n_generated:
        The raw list length handed to :func:`generation_metrics`
        (unparseable strings included).
    n_valid:
        How many parsed via ``Chem.MolFromSmiles`` (mirroring crt/utils.py's
        ``get_mol``) and survived canonicalisation.
    valid_fraction:
        ``n_valid / n_generated`` (0.0 for an empty list).
    n_unique:
        Unique canonical SMILES among the valid ones.
    unique_fraction:
        ``n_unique / n_valid`` (0.0 when nothing parsed).
    n_novel:
        Unique valid canonical SMILES absent from the canonicalised
        reference set -- 0 when no usable reference was supplied.
    novel_fraction:
        ``n_novel / n_unique``; **0.0 when the reference is ``None`` or
        empty** (or entirely unparseable) -- an honest "not measured", not a
        fake 1.0.
    internal_diversity_1:
        IntDiv1: the mean pairwise Tanimoto *distance* (``1 - sim``) over
        the unique valid molecules.  A single unique molecule has no pairs
        and scores 0.0.
    internal_diversity_2:
        IntDiv2: ``sqrt`` of the mean pairwise *squared* distances.
    scaffold_diversity:
        Unique Murcko scaffolds / unique valid molecules.  Note the ratio
        semantics: identical repeats collapse to one unique molecule with
        one scaffold and score 1.0, not 0.0 -- the duplicate collapse is
        what ``unique_fraction`` and IntDiv1 measure; a set of *distinct*
        molecules sharing one scaffold is what pulls this toward 0.0
        (0.0 itself only when nothing parsed).
    n_scaffolds:
        Count of unique Murcko scaffold SMILES -- the acyclic ``''``
        scaffold included (ethanol's), like the ``diversity.py`` buckets.
    pairwise_truncated:
        True when the IntDiv loops ran on the first :data:`PAIRWISE_CAP`
        canonical-sorted unique molecules instead of all of them.
    property_shifts:
        Rows MW / MolLogP / TPSA / QED; columns ``generated_mean``,
        ``reference_mean``, ``shift`` (``generated - reference``); means are
        over each side's unique valid canonical molecules, and the
        ``reference_mean`` / ``shift`` columns are NaN without a usable
        reference.

    Example
    -------
    >>> from pharmaml.generative.moses_metrics import generation_metrics
    >>> audit = generation_metrics(["CCO", "CCO", "not_a_smiles"])
    >>> audit.n_generated, audit.n_valid, audit.n_unique
    (3, 2, 1)
    >>> print(audit)
    MOSES-lite: valid 66.7% | unique 50.0% | novel 0.0% | IntDiv1 0.000 | scaffold diversity 1.000
    """

    n_generated: int
    n_valid: int
    valid_fraction: float
    n_unique: int
    unique_fraction: float
    n_novel: int
    novel_fraction: float
    internal_diversity_1: float
    internal_diversity_2: float
    scaffold_diversity: float
    n_scaffolds: int
    pairwise_truncated: bool
    property_shifts: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        """The property-shift table (the GUI's DataFrame view of the record)."""
        return self.property_shifts

    def __str__(self) -> str:
        """The one-line scorecard: valid / unique / novel %, IntDiv1, scaffold diversity.

        A capped pairwise loop appends its own note, so truncation is never
        silent.
        """
        scorecard = (
            f"MOSES-lite: valid {self.valid_fraction:.1%} | "
            f"unique {self.unique_fraction:.1%} | "
            f"novel {self.novel_fraction:.1%} | "
            f"IntDiv1 {self.internal_diversity_1:.3f} | "
            f"scaffold diversity {self.scaffold_diversity:.3f}"
        )
        if self.pairwise_truncated:
            scorecard += f" | pairwise capped at {PAIRWISE_CAP} unique"
        return scorecard


def _as_smiles_list(smiles: Sequence[str], arg_name: str) -> list[str]:
    """Materialise a SMILES sequence once; reject a bare string.

    Same contract as ``pharmaml.chem.descriptors._as_smiles_list`` (kept
    local instead of importing another module's private helper): a bare
    ``"CCO"`` would otherwise be iterated character-by-character into three
    single-atom "molecules".
    """
    if isinstance(smiles, str):
        preview = smiles if len(smiles) <= 40 else smiles[:40] + "..."
        raise DataValidationError(
            f"{arg_name} must be a list or pandas.Series of SMILES strings, not a "
            f"single string (got {preview!r}); wrap it in a list to compute "
            "one molecule"
        )
    return list(smiles)


def _morgan_bit_matrix(smiles: Sequence[str]) -> np.ndarray:
    """Stack Morgan r=2 / 2048-bit fingerprints as an ``(n, 2048)`` uint8 matrix.

    The modern ``rdFingerprintGenerator.GetMorganGenerator`` (bit-identical
    to ``AllChem.GetMorganFingerprintAsBitVect(radius=2, nBits=2048)``, no
    deprecation warning); the bit vectors are unpacked with
    ``DataStructs.ConvertToNumpyArray`` -- the
    :func:`pharmaml.chem.descriptors.compute_fingerprints` house pattern.
    """
    Chem = require_rdkit()  # actionable DependencyError when RDKit is absent
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS
    )
    matrix = np.zeros((len(smiles), _MORGAN_BITS), dtype=np.uint8)
    for row, one_smiles in enumerate(smiles):
        mol = Chem.MolFromSmiles(one_smiles)
        if mol is None:  # pragma: no cover - canonical SMILES round-trip
            continue
        buffer = np.zeros((_MORGAN_BITS,), dtype=np.uint8)
        # DataStructs.ConvertToNumpyArray fills an existing 1-D array.
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), buffer)
        matrix[row] = buffer
    return matrix


def _internal_diversity(unique_smiles: list[str]) -> tuple[float, float]:
    """(IntDiv1, IntDiv2): mean / RMS pairwise Tanimoto distance.

    The pairwise loop is O(n^2) numpy on the stacked bit matrix:
    :func:`pharmaml.chem.similarity.tanimoto_matrix` (a uint32 matmul)
    gives all similarities in one vectorised call.  Fewer than two
    molecules means no pairs: the honest diversity is 0.0 (MOSES would
    divide by zero here).
    """
    if len(unique_smiles) < 2:
        return 0.0, 0.0
    bits = _morgan_bit_matrix(unique_smiles)
    if bits.shape[0] < 2:  # pragma: no cover - canonical SMILES always parse
        return 0.0, 0.0
    similarities = tanimoto_matrix(bits, bits)
    distances = 1.0 - similarities
    rows, cols = np.triu_indices(distances.shape[0], k=1)
    pairwise = distances[rows, cols]
    return float(pairwise.mean()), float(np.sqrt((pairwise**2).mean()))


def _murcko_scaffold(smiles: str) -> str:
    """Canonical Murcko scaffold SMILES (``''`` when the molecule is acyclic)."""
    Chem = require_rdkit()  # actionable DependencyError when RDKit is absent
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # pragma: no cover - canonical SMILES always parse
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def _descriptor_functions() -> dict[str, Callable[[Any], float]]:
    """The four MOSES property functions (lazy RDKit imports).

    MW / MolLogP / TPSA / QED -- the MOSES distribution-learning properties
    that need no trained model, matching the toolkit's own objective
    implementations (:mod:`pharmaml.generative.scoring`).
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.QED import qed

    return {
        # Descriptors.MolWt / MolLogP are built dynamically inside rdkit's
        # Descriptors module, so no stub sees them (scoring.py's precedent).
        "MW": lambda mol: float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined]
        "MolLogP": lambda mol: float(Descriptors.MolLogP(mol)),  # type: ignore[attr-defined]
        "TPSA": lambda mol: float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": lambda mol: float(qed(mol)),
    }


def _property_means(unique_smiles: list[str]) -> dict[str, float]:
    """Mean of each property over the list (NaN for an empty list)."""
    if not unique_smiles:
        return {name: float("nan") for name in _PROPERTY_ROWS}
    Chem = require_rdkit()  # actionable DependencyError when RDKit is absent
    functions = _descriptor_functions()
    values: dict[str, list[float]] = {name: [] for name in _PROPERTY_ROWS}
    for one_smiles in unique_smiles:
        mol = Chem.MolFromSmiles(one_smiles)
        if mol is None:  # pragma: no cover - canonical SMILES always parse
            continue
        for name, function in functions.items():
            values[name].append(function(mol))
    return {
        name: float(np.mean(vals)) if vals else float("nan")
        for name, vals in values.items()
    }


def _property_shifts(
    generated_unique: list[str], reference_unique: list[str] | None
) -> pd.DataFrame:
    """The property-shift table: generated vs reference means and their shift.

    Rows MW / MolLogP / TPSA / QED, columns ``generated_mean`` /
    ``reference_mean`` / ``shift`` (``generated - reference``).  Without a
    usable reference the reference column -- and therefore the shift --
    is NaN (honest, not a fake 0.0).
    """
    generated = _property_means(generated_unique)
    if reference_unique:
        reference = _property_means(reference_unique)
    else:
        reference = {name: float("nan") for name in _PROPERTY_ROWS}
    return pd.DataFrame(
        {
            "generated_mean": [generated[name] for name in _PROPERTY_ROWS],
            "reference_mean": [reference[name] for name in _PROPERTY_ROWS],
            "shift": [
                generated[name] - reference[name] for name in _PROPERTY_ROWS
            ],
        },
        index=list(_PROPERTY_ROWS),
    )


def generation_metrics(
    generated: Sequence[str], reference: Sequence[str] | None = None
) -> GenerationMetrics:
    """The MOSES-lite audit over a generated SMILES list.

    Valid = parses via ``Chem.MolFromSmiles`` (mirroring crt/utils.py's
    ``get_mol``); canonicalised with ``MolToSmiles`` (``canonic_smiles``);
    unique / novel computed on the canonical forms exactly like
    crt/sampling.py's flow (``set(canonic_smiles)`` then novelty against the
    training set).  ``reference`` ``None`` or empty (or entirely
    unparseable) -> ``novel_fraction`` 0.0 and ``reference_mean`` NaN
    (honest, not a fake 1.0).  Fingerprints: Morgan r=2, 2048 bits,
    Tanimoto.  Scaffold: Murcko
    (``Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol``).

    The pairwise loops (IntDiv1 / IntDiv2) are O(n^2) numpy on stacked bit
    arrays -- ``DataStructs.ConvertToNumpyArray`` stacks the 2048-bit
    vectors into a matrix and
    :func:`pharmaml.chem.similarity.tanimoto_matrix` does the math -- and
    run on at most the first :data:`PAIRWISE_CAP` (2000) canonical-sorted
    unique molecules: a deterministic head, NOT a random subsample, flagged
    by ``pairwise_truncated`` and noted in ``__str__``.  Novelty, scaffold
    diversity and the property shifts always cover *all* the unique valid
    molecules.  Everything is deterministic, so there is no ``random_state``
    parameter.

    Parameters
    ----------
    generated:
        The raw generated SMILES list (invalid strings included -- they are
        what the valid *fraction* denominates).
    reference:
        The reference / training SMILES (canonicalised through the same
        pipeline before comparison; entries that fail to parse are
        dropped).

    Raises
    ------
    DependencyError
        If RDKit is not installed (``pip install 'pharmaml[chem]'``).
    DataValidationError
        If ``generated`` or ``reference`` is a bare string.

    Example
    -------
    >>> audit = generation_metrics(["CC(=O)Oc1ccccc1C(=O)O", "CCO"])
    >>> (audit.n_generated, audit.n_valid, audit.n_unique, audit.n_novel)
    (2, 2, 2, 0)
    """
    generated_list = _as_smiles_list(generated, "generated")
    reference_list = (
        None if reference is None else _as_smiles_list(reference, "reference")
    )

    # -- the crt flow: parse + canonicalise, drop the unparseable ---------
    canonical_forms = [canonical_smiles(smiles) for smiles in generated_list]
    valid = [smiles for smiles in canonical_forms if smiles is not None]
    n_generated = len(generated_list)
    n_valid = len(valid)
    valid_fraction = n_valid / n_generated if n_generated else 0.0

    # canonical-sorted: the pairwise cap's deterministic head
    unique = sorted(set(valid))
    n_unique = len(unique)
    unique_fraction = n_unique / n_valid if n_valid else 0.0

    # -- novelty against the canonicalised reference (check_novelty) -----
    reference_canonical: set[str] = set()
    if reference_list:
        reference_canonical = {
            smiles
            for smiles in (canonical_smiles(s) for s in reference_list)
            if smiles is not None
        }
    if reference_canonical:
        n_novel = sum(1 for smiles in unique if smiles not in reference_canonical)
        novel_fraction = n_novel / n_unique if n_unique else 0.0
    else:
        # no usable reference: novelty is *not measured* -- an honest 0.0
        # (check_novelty's empty-training-set branch does the same), and
        # the reference property column goes NaN below.
        n_novel = 0
        novel_fraction = 0.0

    # -- internal diversity (capped pairwise loops) -----------------------
    pairwise_truncated = len(unique) > PAIRWISE_CAP
    int_div_1, int_div_2 = _internal_diversity(unique[:PAIRWISE_CAP])
    if pairwise_truncated:
        _LOG.info(
            "Pairwise diversity capped at %d of %d unique molecules "
            "(deterministic canonical-sorted head)",
            PAIRWISE_CAP,
            len(unique),
        )

    # -- scaffold diversity ------------------------------------------------
    scaffolds = [_murcko_scaffold(smiles) for smiles in unique]
    n_scaffolds = len(set(scaffolds))
    scaffold_diversity = n_scaffolds / n_unique if n_unique else 0.0

    # -- property distribution shifts --------------------------------------
    reference_unique = sorted(reference_canonical) if reference_canonical else None
    shifts = _property_shifts(unique, reference_unique)

    audit = GenerationMetrics(
        n_generated=n_generated,
        n_valid=n_valid,
        valid_fraction=float(valid_fraction),
        n_unique=n_unique,
        unique_fraction=float(unique_fraction),
        n_novel=n_novel,
        novel_fraction=float(novel_fraction),
        internal_diversity_1=int_div_1,
        internal_diversity_2=int_div_2,
        scaffold_diversity=float(scaffold_diversity),
        n_scaffolds=n_scaffolds,
        pairwise_truncated=pairwise_truncated,
        property_shifts=shifts,
    )
    _LOG.info("MOSES-lite audit of %d generated SMILES: %s", n_generated, audit)
    return audit

"""Synthetic Accessibility (SA) scoring -- Ertl & Schuffenhauer (2009).

MASTER_PROMPT part3 section 24: *"Integrate Synthetic Accessibility (SA)
scoring into the existing generative module... Use RDKit's established
SA_Score implementation when compatible and available"* with the
methodology reference *"Estimation of synthetic accessibility score of
drug-like molecules based on molecular complexity and fragment
contributions", Peter Ertl and Ansgar Schuffenhauer, Journal of
Cheminformatics 1:8 (2009)* and the official implementation at
``github.com/rdkit/rdkit/tree/master/Contrib/SA_Score``.

The score is the established three-part Ertl estimate:

* **fragment score** -- the average log-frequency contribution of the
  molecule's radius-2 Morgan sparse count fingerprint bits against a
  fragment model built from ~1 M reaction-product molecules (the
  Fsynth database; unknown bits contribute the floor ``-4``);
* **complexity penalty** -- size (``n_atoms**1.005 - n_atoms``),
  stereo centres, spiro atoms, bridgehead atoms and macrocycles
  (each ``log10(n + 1)``; the macrocycle term is the implementation's
  fixed ``log10(2)`` when any ring larger than 8 exists -- a documented
  deviation from the paper that behaves better with 2+ macrocycles);
* **symmetry correction** -- ``0.5 * ln(n_atoms / distinct_bits)`` when
  the fingerprint collapses (highly symmetrical molecules are *easier*
  to synthesise; added in the reference implementation's v1.1).

The raw value is affine-mapped into the conventional ``[1, 10]`` range
with the ``[-4, 2.5]`` clipping window and the ``>8`` logarithmic
softening, exactly as the reference implementation does:

.. note::
    **Direction**: *lower = estimated easier to synthesise*, higher =
    estimated more difficult.  The conventional drug-like band is
    roughly 1-4; scores above ~6 signal serious complexity.  The score
    is a *computational estimate on molecular complexity and fragment
    familiarity* -- it is **not** a synthesis route prediction and must
    never be presented as proof that a molecule can (or cannot) be
    made.  See :data:`SA_SCORE_LIMITATIONS`.

**Provenance of the fragment model**: the ~617-entry bit->score table
is *not vendored* into this package.  It is read, on first use, from
the ``fpscores.pkl.gz`` database that ships inside the RDKit wheel
itself (``rdkit/Contrib/SA_Score/``) -- the same data file the official
``sascorer.py`` Contrib script loads.  When the file is absent (an
RDKit build without Contrib), a :class:`~pharmaml.exceptions.DependencyError`
names the exact path that was tried.  The fingerprint generation uses
RDKit's own ``rdFingerprintGenerator.GetMorganGenerator(radius=2)`` so
bit ids match the model exactly.

Public surface (part3 section 24.2-24.4):

* :func:`sa_score` -- one SMILES (or mol) -> :class:`SAScoreResult`;
* :func:`sa_scores` -- batch with per-molecule validity accounting;
* :func:`sa_score_frame` -- the §24.3 results table (sortable/filterable);
* :func:`filter_by_sa` -- §24.4 candidate filtering that *preserves* the
  rejected candidates (nothing is silently discarded);
* :func:`sa_score_summary` -- direction-aware summary statistics for a
  batch (the interface's "score distribution" row).

Examples
--------
>>> from pharmaml.chem.sa_scoring import sa_score, sa_scores
>>> r = sa_score("CC(=O)Oc1ccccc1C(=O)O")     # aspirin -- drug-like
>>> r.valid and 1.0 <= r.score <= 4.0
True
>>> abs(r.score - 1.58) < 0.02                # the reference value
True
>>> bad = sa_score("not-a-smiles")
>>> bad.valid
False
>>> bad.score is None
True
>>> batch = sa_scores(["CCO", "not-a-smiles", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"])
>>> (batch.n_scored, batch.n_invalid)
(2, 1)
>>> complex_mol = sa_score(
...     "CC1=C(C[C@@H]2[C@]1(C[C@H]([C@@H](C2=O)O)NC(=O)c3ccccc3)O)OC(=O)C")
>>> simple = sa_score("CCO")
>>> complex_mol.score > simple.score        # direction: harder -> higher
True
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

__all__ = [
    "SA_SCORE_LIMITATIONS",
    "SA_SCORE_METHOD",
    "SAFilterResult",
    "SAScoreBatch",
    "SAScoreResult",
    "filter_by_sa",
    "sa_score",
    "sa_score_frame",
    "sa_score_summary",
    "sa_scores",
]

_LOG = get_logger(__name__)

#: Citation shown in every result's provenance string.
SA_SCORE_METHOD = (
    "Ertl-Schuffenhauer synthetic accessibility score "
    "(J Cheminform 1:8, 2009; RDKit Contrib SA_Score implementation)"
)

#: The interface copy for §24.5's "document its limitations" requirement.
SA_SCORE_LIMITATIONS = (
    "The SA score is a computational estimate from molecular complexity "
    "and fragment familiarity. It does not plan or verify a synthesis "
    "route, does not consider available starting materials or reaction "
    "conditions, and a favourable score is not evidence that a molecule "
    "can actually be synthesised."
)

#: Affine-scaling window of the reference implementation (v1.1).
_RAW_MIN = -4.0
_RAW_MAX = 2.5

#: Fragment-score floor for fingerprint bits missing from the model.
_UNKNOWN_BIT_SCORE = -4.0

#: Lazy caches: {bit_id: fragment score} and the loaded marker.
_FRAGMENT_SCORES: dict[int, float] | None = None


def _fragment_scores() -> dict[int, float]:
    """Load (and cache) the Ertl fragment model from the RDKit wheel.

    The database is RDKit's own ``Contrib/SA_Score/fpscores.pkl.gz`` --
    the gzip pickle the official ``sascorer.py`` reads.  The pickle is a
    list of ``[score, bit_id, bit_id, ...]`` rows which this loader
    flattens into the ``{bit_id: score}`` mapping the scorer needs.
    """
    global _FRAGMENT_SCORES
    if _FRAGMENT_SCORES is not None:
        return _FRAGMENT_SCORES
    try:
        import rdkit
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for SA scoring. From the toolkit package root, run: "
            "python -m pip install -e '.[chem]'"
        ) from exc
    data_path = Path(rdkit.__file__).parent / "Contrib" / "SA_Score" / "fpscores.pkl.gz"
    if not data_path.exists():
        raise DependencyError(
            "The RDKit Contrib SA_Score database was not found at "
            f"'{data_path}'. This RDKit build ships without the Contrib "
            "tree; install the full 'rdkit' wheel to enable SA scoring."
        )
    with gzip.open(data_path, "rb") as handle:
        rows = pickle.load(handle)
    table: dict[int, float] = {}
    for row in rows:
        score = float(row[0])
        for bit in row[1:]:
            table[int(bit)] = score
    _FRAGMENT_SCORES = table
    _LOG.debug("loaded %d SA fragment constants from %s", len(table), data_path)
    return table


def _ertl_raw_score(mol: Any) -> float:
    """Replicate ``sascorer.calculateScore``'s raw pre-scaling value.

    Kept separate from the [1, 10] affine map so the two halves of the
    method stay independently readable (and independently testable).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

    fscores = _fragment_scores()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sparse = generator.GetSparseCountFingerprint(mol)

    # fragment score: mean log-frequency contribution over all bit counts
    score1 = 0.0
    n_bits_total = 0
    nonzero = sparse.GetNonzeroElements()
    for bit, count in nonzero.items():
        n_bits_total += count
        score1 += fscores.get(bit, _UNKNOWN_BIT_SCORE) * count
    score1 /= n_bits_total

    # complexity penalty: size, stereo, spiro, bridgehead, macrocycle
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_macrocycles = sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgehead + 1)
    # the implementation deviates from the paper on purpose: a *fixed*
    # log10(2) whenever any macrocycle exists behaves better with 2+
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0
    score2 = (
        -size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    # symmetry (fingerprint-density) correction, reference v1.1
    score3 = 0.0
    distinct_bits = len(nonzero)
    if n_atoms > distinct_bits:
        score3 = math.log(n_atoms / distinct_bits) * 0.5

    return score1 + score2 + score3


def _rescale(raw: float) -> float:
    """The reference implementation's affine map into ``[1, 10]``."""
    value = 11.0 - (raw - _RAW_MIN + 1.0) / (_RAW_MAX - _RAW_MIN) * 9.0
    if value > 8.0:
        value = 8.0 + math.log(value + 1.0 - 9.0)
    if value > 10.0:
        return 10.0
    if value < 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class SAScoreResult:
    """One molecule's SA score with full validation context (§24.2).

    ``score`` is ``None`` exactly when ``valid`` is ``False`` -- an
    invalid SMILES never gets a misleading number, and the original
    string is preserved for debugging (§22.4's rule, applied here).
    """

    #: The SMILES as given (never rewritten on failure).
    smiles: str
    #: The estimated SA score in [1, 10]; ``None`` when invalid.
    score: float | None
    #: Whether the SMILES parsed to a non-empty valid molecule.
    valid: bool
    #: Human-readable reason when ``valid`` is ``False``.
    error: str | None = None
    #: Canonical SMILES when the molecule parsed (``None`` otherwise).
    canonical_smiles: str | None = None
    #: Provenance: method + data citation (never a bare number).
    method: str = SA_SCORE_METHOD

    @property
    def direction(self) -> str:
        """Score-direction copy for interfaces (§24.2 requirement)."""
        return "lower = estimated easier to synthesise"

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly row for report tables and the web bridge."""
        return {
            "smiles": self.smiles,
            "score": self.score,
            "valid": self.valid,
            "error": self.error,
            "canonicalSmiles": self.canonical_smiles,
            "method": self.method,
            "direction": self.direction,
        }


def sa_score(source: str | Any) -> SAScoreResult:
    """Score one SMILES string (or an RDKit molecule).

    Parameters
    ----------
    source:
        Either a SMILES string or an already-parsed ``rdkit.Chem.Mol``.
        Strings are parsed safely; invalid input returns a result with
        ``valid=False`` and ``score=None`` -- never an exception.

    Raises
    ------
    pharmaml.exceptions.DependencyError
        When RDKit (or its Contrib SA_Score database) is unavailable.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for SA scoring. From the toolkit package root, run: "
            "python -m pip install -e '.[chem]'"
        ) from exc

    if not isinstance(source, str):
        mol = source
        given = Chem.MolToSmiles(mol) if mol is not None else ""
    else:
        given = source
        if not given.strip():
            return SAScoreResult(
                smiles=given, score=None, valid=False,
                error="empty SMILES string",
            )
        # RDKit logs parse errors to stderr before returning None; the
        # result's ``error`` field is the message users actually read, so
        # the raw parser trace is silenced (the docking-prep precedent).
        from rdkit.rdBase import DisableLog, EnableLog

        DisableLog("rdApp.error")
        try:
            mol = Chem.MolFromSmiles(given)
        finally:
            EnableLog("rdApp.error")

    if mol is None or mol.GetNumAtoms() == 0:
        return SAScoreResult(
            smiles=given, score=None, valid=False,
            error="SMILES did not parse to a valid molecule",
        )
    raw = _ertl_raw_score(mol)
    return SAScoreResult(
        smiles=given,
        score=round(_rescale(raw), 3),
        valid=True,
        canonical_smiles=Chem.MolToSmiles(mol),
    )


@dataclass(frozen=True)
class SAScoreBatch:
    """Batch results with the validity accounting (§24.2/24.3)."""

    results: tuple[SAScoreResult, ...]
    elapsed_seconds: float = 0.0

    @property
    def n_scored(self) -> int:
        return sum(1 for r in self.results if r.valid)

    @property
    def n_invalid(self) -> int:
        return sum(1 for r in self.results if not r.valid)

    @property
    def valid_scores(self) -> list[float]:
        return [r.score for r in self.results if r.valid and r.score is not None]

    def sorted_by_score(self, descending: bool = False) -> list[SAScoreResult]:
        """§24.3's sorting: easiest-to-synthesise first by default."""
        return sorted(
            (r for r in self.results if r.valid),
            key=lambda r: r.score if r.score is not None else 0.0,
            reverse=descending,
        )


def sa_scores(smiles: Iterable[str]) -> SAScoreBatch:
    """Score a batch of SMILES with per-molecule validity (§24.2).

    Invalid entries never abort the batch: each gets its own
    ``valid=False`` result and the accounting exposes ``n_invalid``.
    """
    import time

    start = time.perf_counter()
    results = tuple(sa_score(s) for s in smiles)
    return SAScoreBatch(
        results=results, elapsed_seconds=round(time.perf_counter() - start, 4)
    )


def sa_score_frame(smiles: Iterable[str]) -> pd.DataFrame:
    """The §24.3 generative-results table: SMILES, SA, validity, order.

    Sort/filter downstream with plain pandas; ``score`` is ``NaN`` for
    invalid rows so sorting never raises.
    """
    batch = sa_scores(smiles)
    rows = [
        {
            "index": i,
            "smiles": r.smiles,
            "canonical_smiles": r.canonical_smiles,
            "sa_score": r.score if r.valid else float("nan"),
            "valid": r.valid,
            "error": r.error,
        }
        for i, r in enumerate(batch.results)
    ]
    return pd.DataFrame(
        rows,
        columns=["index", "smiles", "canonical_smiles", "sa_score", "valid", "error"],
    )


@dataclass(frozen=True)
class SAFilterResult:
    """§24.4 filtering outcome: kept AND rejected are both preserved."""

    kept: tuple[SAScoreResult, ...] = field(default_factory=tuple)
    rejected: tuple[SAScoreResult, ...] = field(default_factory=tuple)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def filter_by_sa(
    results: Sequence[SAScoreResult],
    min_sa: float | None = None,
    max_sa: float | None = None,
) -> SAFilterResult:
    """Filter scored candidates by an inclusive SA range (§24.4).

    * ``max_sa`` -- the "maximum acceptable SA score" constraint;
    * ``min_sa`` -- the "minimum required score" constraint;
    * both ``None`` -- no SA filtering (nothing is filtered: every
      candidate, including not-yet-scored invalid rows, stays in
      ``kept`` -- the spec's "do not permanently discard generated
      candidates unless the user explicitly enables such filtering").

    Invalid rows are *rejected* (they carry no score to compare) only
    when a constraint is actually active, and never discarded: they
    land in ``rejected`` with their original SMILES and error, ready
    for later analysis -- "preserve rejected candidate information".
    """
    if min_sa is not None and max_sa is not None and min_sa > max_sa:
        raise DataValidationError(
            f"min_sa ({min_sa}) must not exceed max_sa ({max_sa})"
        )
    if min_sa is None and max_sa is None:
        return SAFilterResult(kept=tuple(results), rejected=())
    kept: list[SAScoreResult] = []
    rejected: list[SAScoreResult] = []
    for r in results:
        if not r.valid or r.score is None:
            rejected.append(r)
            continue
        if min_sa is not None and r.score < min_sa:
            rejected.append(r)
            continue
        if max_sa is not None and r.score > max_sa:
            rejected.append(r)
            continue
        kept.append(r)
    return SAFilterResult(kept=tuple(kept), rejected=tuple(rejected))


def sa_score_summary(smiles: Iterable[str]) -> dict[str, Any]:
    """Direction-aware batch statistics for interfaces (§24.3).

    Only *actual computed values* are reported (the spec's rule): every
    statistic is over the valid subset, and the payload names the
    method, the direction and the limitations so no interface can
    present the score without its context.
    """
    batch = sa_scores(smiles)
    scores = batch.valid_scores
    summary: dict[str, Any] = {
        "n_total": len(batch.results),
        "n_scored": batch.n_scored,
        "n_invalid": batch.n_invalid,
        "method": SA_SCORE_METHOD,
        "direction": "lower = estimated easier to synthesise",
        "limitations": SA_SCORE_LIMITATIONS,
        "elapsed_seconds": batch.elapsed_seconds,
    }
    if scores:
        import numpy as np

        arr = np.asarray(scores, dtype=float)
        summary.update(
            {
                "min": round(float(arr.min()), 3),
                "max": round(float(arr.max()), 3),
                "mean": round(float(arr.mean()), 3),
                "median": round(float(np.median(arr)), 3),
                "std": round(float(arr.std(ddof=1)) if len(scores) > 1 else 0.0, 3),
            }
        )
    return summary

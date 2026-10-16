"""Objective library for molecule generation: SMILES -> score in [0, 1].

Every optimiser in :mod:`pharmaml.generative` (genetic algorithm and
reinforcement learning alike) maximises a *scoring objective* -- a callable
mapping a SMILES string to a float in ``[0, 1]``.  This module ships the
common pharmacology-flavoured objectives:

* :class:`QedObjective` -- RDKit quantitative estimate of drug-likeness;
* :class:`LogPBandObjective` / :class:`MolecularWeightBandObjective` /
  :class:`TpsaBandObjective` -- Gaussian bands around a target property
  window (the interval ``[lo, hi]`` is the ±1σ band: 1.0 at the centre,
  ≈0.61 at the edges, smoothly decaying outside);
* :class:`SimilarityObjective` -- Morgan-fingerprint Tanimoto similarity
  to a seed molecule;
* :class:`NoveltyObjective` -- structural novelty against a reference
  corpus (``1 - max`` Tanimoto to any corpus member -- a corpus member
  scores ``0.0``, a structurally distant molecule scores high);
* :class:`SubstructureObjective` -- hard pharmacophore constraint (1.0 when
  the SMARTS matches, 0.0 otherwise);
* :class:`MultiObjective` -- weighted geometric mean of several objectives
  with zero-score vetoing, so one hard constraint can reject a molecule
  outright.

Invalid SMILES never crash an objective: they score ``0.0`` and are
counted (``n_invalid`` / ``n_scored`` / :meth:`ScoringObjective.reset_stats`)
-- the accounting the optimisers use to report valid-generation fractions.

Custom objectives subclass :class:`ScoringObjective` and implement
:meth:`ScoringObjective.score`; using the :meth:`_parse` helper gives the
invalid-input accounting for free::

    from pharmaml.generative import ScoringObjective

    class HeavyAtomObjective(ScoringObjective):
        name = "heavy_atoms"
        def score(self, smiles: str) -> float:
            mol = self._parse(smiles)
            if mol is None:
                return 0.0
            return min(mol.GetNumHeavyAtoms() / 30.0, 1.0)

RDKit is imported lazily (the :mod:`pharmaml.chem` precedent): importing
this module never requires it, and calling ``score`` without RDKit raises
an actionable :class:`~pharmaml.exceptions.DependencyError`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import require_rdkit
from .corpus import DEFAULT_CORPUS

__all__ = [
    "LogPBandObjective",
    "MolecularWeightBandObjective",
    "MultiObjective",
    "NoveltyObjective",
    "QedObjective",
    "ScoringObjective",
    "SimilarityObjective",
    "SubstructureObjective",
    "TpsaBandObjective",
]

_LOG = get_logger(__name__)

#: Smallest band half-width accepted by the band objectives (a zero-width
#: band would make the Gaussian explode; this floor keeps it finite).
_MIN_BAND_WIDTH = 1e-6


def _band_score(value: float, lo: float, hi: float) -> float:
    """Gaussian band score: 1.0 at the centre, ≈0.61 at the edges."""
    centre = (lo + hi) / 2.0
    sigma = max((hi - lo) / 2.0, _MIN_BAND_WIDTH)
    z = (value - centre) / sigma
    return math.exp(-0.5 * z * z)


class ScoringObjective:
    """Base class for generation objectives: ``score(smiles) -> [0, 1]``.

    Subclasses set the ``name`` class attribute (used as the breakdown
    column name inside :class:`MultiObjective`) and implement
    :meth:`score`.  Parsing a SMILES through :meth:`_parse` records the
    invalid-input accounting (``n_scored`` / ``n_invalid``) that the
    optimisers surface in their history.

    Example
    -------
    >>> from pharmaml.generative import ScoringObjective
    >>> class ConstantObjective(ScoringObjective):
    ...     name = "constant"
    ...     def score(self, smiles):
    ...         return 0.5
    """

    #: Short identifier used in breakdown tables and result columns.
    name: str = "objective"

    def __init__(self) -> None:
        self._n_scored = 0
        self._n_invalid = 0

    # -- accounting ------------------------------------------------------
    @property
    def n_scored(self) -> int:
        """Number of SMILES scored since construction / last reset."""
        return self._n_scored

    @property
    def n_invalid(self) -> int:
        """Number of unparseable SMILES seen (each scored 0.0)."""
        return self._n_invalid

    def reset_stats(self) -> None:
        """Zero the ``n_scored`` / ``n_invalid`` counters."""
        self._n_scored = 0
        self._n_invalid = 0

    # -- contract --------------------------------------------------------
    def _parse(self, smiles: str) -> Any:
        """Parse ``smiles`` with RDKit, counting invalid inputs.

        Returns the RDKit ``Mol`` or ``None`` (invalid / blank / non-str).
        """
        Chem = require_rdkit()  # actionable DependencyError when RDKit is absent
# rdkit-stubs does not re-export DisableLog/EnableLog on rdkit.RDLogger,
        # but rdBase (their defining module) is fully typed.
        from rdkit.rdBase import DisableLog, EnableLog

        self._n_scored += 1
        if not isinstance(smiles, str) or not smiles.strip():
            self._n_invalid += 1
            return None
        # RDKit logs parse errors to stderr before returning None; the
        # accounting below is friendlier than the raw parser trace, and the
        # optimisers parse thousands of candidates per run.
        DisableLog("rdApp.error")
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:  # noqa: BLE001 - RDKit can raise on hostile input
            mol = None
        finally:
            EnableLog("rdApp.error")
        if mol is None:
            self._n_invalid += 1
        return mol

    def score(self, smiles: str) -> float:
        """Return the objective value of ``smiles`` in ``[0, 1]``.

        Invalid SMILES score ``0.0`` (counted in ``n_invalid``) instead of
        raising -- generation loops feed raw sampled/mutated strings here.
        """
        raise NotImplementedError

    def score_many(self, smiles_list: Sequence[str]) -> list[float]:
        """Score a batch of SMILES (convenience loop over :meth:`score`)."""
        return [self.score(smiles) for smiles in smiles_list]

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return f"{type(self).__name__}(name={self.name!r})"


class QedObjective(ScoringObjective):
    """RDKit QED (quantitative estimate of drug-likeness).

    QED blends eight desirability functions (molecular weight, LogP, TPSA,
    rotatable bonds, aromatic rings, alerts, ...) into a single ``[0, 1]``
    score -- 1.0 is a maximally drug-like molecule.

    Example
    -------
    >>> QedObjective().score("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
    0.55...
    """

    name = "qed"

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        from rdkit.Chem.QED import qed

        try:
            return float(min(max(qed(mol), 0.0), 1.0))
        except Exception:  # noqa: BLE001 - exotic valences can upset QED
            _LOG.warning("QED failed on %r -- scoring 0.0", smiles)
            return 0.0


class _BandObjective(ScoringObjective):
    """Shared machinery for the property-band objectives."""

    def __init__(self, lo: float, hi: float, what: str) -> None:
        super().__init__()
        for bound, label in ((lo, "lo"), (hi, "hi")):
            if (
                not isinstance(bound, (int, float))
                or isinstance(bound, bool)
                or not math.isfinite(bound)
            ):
                raise DataValidationError(
                    f"{what} band bound {label}={bound!r} must be a finite number"
                )
        if lo > hi:
            raise DataValidationError(
                f"{what} band is empty: lo={lo} is greater than hi={hi}"
            )
        if hi - lo < 2 * _MIN_BAND_WIDTH:
            raise DataValidationError(
                f"{what} band is too narrow: hi - lo = {hi - lo:.3g} "
                f"(need > {2 * _MIN_BAND_WIDTH:.3g})"
            )
        self.lo = float(lo)
        self.hi = float(hi)

    def _descriptor(self, mol: Any) -> float:
        raise NotImplementedError

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        return _band_score(self._descriptor(mol), self.lo, self.hi)


class LogPBandObjective(_BandObjective):
    """Gaussian band on Crippen LogP (wild octanol/water behaviour excluded).

    The interval ``[lo, hi]`` is the ±1σ band around the target: a molecule
    exactly at the centre scores 1.0, at either edge ≈0.61, and the score
    decays smoothly further out -- so molecules *near* the window are
    still rewarded.

    Example
    -------
    >>> round(LogPBandObjective(0.31, 2.31).score("CC(=O)Oc1ccccc1C(=O)O"), 6)
    1.0
    """

    name = "logp_band"

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__(lo, hi, what="LogP")

    def _descriptor(self, mol: Any) -> float:
        from rdkit.Chem import Descriptors

        # rdkit-stubs omits the dynamically-generated descriptor functors
        # on rdkit.Chem.Descriptors (MolLogP among them); the runtime API
        # is stable and documented.
        return float(Descriptors.MolLogP(mol))  # type: ignore[attr-defined]


class MolecularWeightBandObjective(_BandObjective):
    """Gaussian band on average molecular weight (Da).  See
    :class:`LogPBandObjective` for the band semantics."""

    name = "mw_band"

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__(lo, hi, what="molecular weight")
        if self.lo <= 0:
            raise DataValidationError(
                f"molecular weight band must be positive, got lo={self.lo}"
            )

    def _descriptor(self, mol: Any) -> float:
        from rdkit.Chem import Descriptors

        # rdkit-stubs omits the dynamically-generated descriptor functors
        # on rdkit.Chem.Descriptors (MolWt among them); the runtime API is
        # stable and documented.
        return float(Descriptors.MolWt(mol))  # type: ignore[attr-defined]


class TpsaBandObjective(_BandObjective):
    """Gaussian band on topological polar surface area (Å²).  See
    :class:`LogPBandObjective` for the band semantics."""

    name = "tpsa_band"

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__(lo, hi, what="TPSA")
        if self.lo <= 0:
            raise DataValidationError(
                f"TPSA band must be positive, got lo={self.lo}"
            )

    def _descriptor(self, mol: Any) -> float:
        from rdkit.Chem import rdMolDescriptors

        return float(rdMolDescriptors.CalcTPSA(mol))


class SimilarityObjective(ScoringObjective):
    """Tanimoto similarity (Morgan fingerprints) to one seed molecule.

    Parameters
    ----------
    seed_smiles:
        The reference molecule.  Must be a parseable SMILES.
    radius:
        Morgan fingerprint radius (default 2, the "ECFP4" convention).
    n_bits:
        Fingerprint bit count (default 2048).

    Raises
    ------
    DataValidationError
        If ``seed_smiles`` cannot be parsed.

    Example
    -------
    >>> SimilarityObjective("CC(=O)Oc1ccccc1C(=O)O").score(
    ...     "CC(=O)Oc1ccccc1C(=O)O")   # aspirin vs itself
    1.0
    """

    name = "similarity"

    def __init__(self, seed_smiles: str, radius: int = 2, n_bits: int = 2048) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.rdBase import DisableLog, EnableLog

        if not isinstance(seed_smiles, str) or not seed_smiles.strip():
            raise DataValidationError("seed_smiles must be a non-empty string")
        # Silence the raw parser trace: the DataValidationError below is the
        # actionable message (the docking/ligands._validate_smiles precedent).
        DisableLog("rdApp.error")
        try:
            mol = Chem.MolFromSmiles(seed_smiles)
        finally:
            EnableLog("rdApp.error")
        if mol is None:
            raise DataValidationError(
                f"seed_smiles {seed_smiles!r} is not a parseable SMILES"
            )
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise DataValidationError(
                f"radius must be a non-negative int, got {radius!r}"
            )
        if not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits < 32:
            raise DataValidationError(f"n_bits must be an int >= 32, got {n_bits!r}")
        # The modern generator API: bit-identical to the classic
        # ``GetMorganFingerprintAsBitVect`` but without its per-call
        # deprecation warning (generation loops score thousands of molecules).
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self.seed_smiles = Chem.MolToSmiles(mol)
        self.radius = radius
        self.n_bits = n_bits
        self._seed_fp = self._generator.GetFingerprint(mol)

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        from rdkit import DataStructs

        fp = self._generator.GetFingerprint(mol)
        return float(DataStructs.TanimotoSimilarity(fp, self._seed_fp))


class NoveltyObjective(ScoringObjective):
    """Structural novelty against a reference corpus.

    ``score = 1 - max(Tanimoto(candidate, m) for m in corpus)`` computed on
    Morgan fingerprints: a molecule already in the corpus is by definition
    *not* novel (score ``0.0``), while a molecule far from every corpus
    member scores towards ``1.0``.  Used inside a :class:`MultiObjective`
    the objective acts as a soft anti-rediscovery constraint: one corpus
    look-alike vetoes the whole molecule.

    Parameters
    ----------
    corpus:
        Reference SMILES (default: :data:`~pharmaml.generative.DEFAULT_CORPUS`).
        Entries are canonicalised and de-duplicated; invalid entries are
        dropped with a warning.
    radius:
        Morgan fingerprint radius (default 2, the "ECFP4" convention).
    n_bits:
        Fingerprint bit count (default 2048).

    Raises
    ------
    DataValidationError
        If ``corpus`` is not a sequence of SMILES, is empty, holds no
        parseable entry, or ``radius`` / ``n_bits`` are out of range.

    Example
    -------
    >>> NoveltyObjective(corpus=["CCO"]).score("CCO")   # corpus member
    0.0
    >>> round(NoveltyObjective(corpus=["CC(=O)Oc1ccccc1C(=O)O"]).score(
    ...     "CC(=O)Nc1ccc(O)cc1"), 3)   # paracetamol vs aspirin-only corpus
    0.778
    """

    name = "novelty"

    def __init__(
        self,
        corpus: Sequence[str] | None = None,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.rdBase import DisableLog, EnableLog

        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise DataValidationError(
                f"radius must be a non-negative int, got {radius!r}"
            )
        if not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits < 32:
            raise DataValidationError(f"n_bits must be an int >= 32, got {n_bits!r}")

        if corpus is None:
            entries: list[str] = list(DEFAULT_CORPUS)
        elif isinstance(corpus, str):
            # iterating a bare string would treat CHARACTERS as molecules
            raise DataValidationError(
                "corpus must be a sequence of SMILES strings, not a single "
                f"string (got {corpus!r} of length {len(corpus)})"
            )
        else:
            try:
                entries = list(corpus)
            except TypeError as exc:
                raise DataValidationError(
                    "corpus must be a sequence of SMILES strings "
                    f"(got {type(corpus).__name__})"
                ) from exc
        n_non_string = sum(1 for entry in entries if not isinstance(entry, str))
        entries = [entry for entry in entries if isinstance(entry, str)]
        if n_non_string:
            # the docstring contract: invalid entries are dropped WITH A
            # WARNING (unparseable strings warn below; non-strings must not
            # vanish silently either -- silent data loss is the failure mode
            # this package refuses).
            _LOG.warning(
                "dropping %d non-string corpus entr(ies) -- corpus entries "
                "must be SMILES strings",
                n_non_string,
            )
        if not entries:
            raise DataValidationError(
                "corpus is empty (or holds no string entries) -- pass a "
                "non-empty list of SMILES strings"
                + (
                    f" ({n_non_string} non-string entr(ies) dropped)"
                    if n_non_string
                    else ""
                )
            )

        canonical: list[str] = []
        # RDKit logs parse errors to stderr before returning None; invalid
        # corpus entries are dropped with a warning instead (the RL prior's
        # corpus-sanitisation precedent).
        DisableLog("rdApp.error")
        try:
            for entry in entries:
                if not entry.strip():
                    _LOG.warning("dropping blank corpus SMILES %r", entry)
                    continue
                mol = Chem.MolFromSmiles(entry)
                if mol is None:
                    _LOG.warning("dropping invalid corpus SMILES %r", entry)
                    continue
                canonical.append(Chem.MolToSmiles(mol))
        finally:
            EnableLog("rdApp.error")
        canonical = list(dict.fromkeys(canonical))
        if not canonical:
            raise DataValidationError(
                "corpus holds no valid SMILES -- every entry failed to parse"
            )

        self.radius = radius
        self.n_bits = n_bits
        self.corpus = tuple(canonical)
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self._corpus_fps = [
            self._generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            for smiles in canonical
        ]

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        from rdkit import DataStructs

        fp = self._generator.GetFingerprint(mol)
        return float(1.0 - max(DataStructs.BulkTanimotoSimilarity(fp, self._corpus_fps)))


class SubstructureObjective(ScoringObjective):
    """Hard pharmacophore constraint: 1.0 when ``smarts`` matches.

    Raises
    ------
    DataValidationError
        If ``smarts`` cannot be parsed as a SMARTS query.

    Example
    -------
    >>> SubstructureObjective("c1ccccc1").score(
    ...     "CC(=O)Oc1ccccc1C(=O)O")   # aspirin contains benzene
    1.0
    """

    name = "substructure"

    def __init__(self, smarts: str) -> None:
        super().__init__()
        Chem = require_rdkit()
        from rdkit.rdBase import DisableLog, EnableLog

        if not isinstance(smarts, str) or not smarts.strip():
            raise DataValidationError("smarts must be a non-empty string")
        DisableLog("rdApp.error")
        try:
            query = Chem.MolFromSmarts(smarts)
        finally:
            EnableLog("rdApp.error")
        if query is None:
            raise DataValidationError(f"smarts {smarts!r} is not a parseable SMARTS")
        self.smarts = smarts
        self._query = query

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        return 1.0 if mol.HasSubstructMatch(self._query) else 0.0


class MultiObjective(ScoringObjective):
    """Weighted geometric mean of several objectives, with veto semantics.

    ``total = prod(score_i ** weight_i) ** (1 / sum(weight_i))`` -- a
    product of per-objective scores, so **any zero score vetoes the whole
    molecule** (one hard constraint can reject what the others love).
    Geometric (not arithmetic) weighting keeps the total in ``[0, 1]`` and
    rewards balanced solutions.

    Parameters
    ----------
    objectives:
        Two or more objectives (any :class:`ScoringObjective`, or any
        object exposing ``name`` and a callable ``score``).
    weights:
        Positive weights, one per objective (default: all 1.0).

    Raises
    ------
    DataValidationError
        On empty / single-objective lists, mismatched weights, or
        non-positive / non-finite weights.

    Example
    -------
    >>> obj = MultiObjective(
    ...     [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[1.0, 1.0])
    >>> total, breakdown = obj.score_with_breakdown(
    ...     "CC(=O)Oc1ccccc1C(=O)O")
    >>> sorted(breakdown)
    ['logp_band', 'qed']
    """

    name = "multi"

    def __init__(
        self,
        objectives: Sequence[Any],
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        objectives = list(objectives)
        if len(objectives) < 2:
            raise DataValidationError(
                "MultiObjective needs at least two objectives "
                f"(got {len(objectives)}) -- a single objective needs no combinator"
            )
        for i, objective in enumerate(objectives):
            if not hasattr(objective, "score") or not callable(objective.score):
                raise DataValidationError(
                    f"objective #{i} ({objective!r}) has no callable score() method"
                )
            if not hasattr(objective, "name"):
                raise DataValidationError(
                    f"objective #{i} ({objective!r}) has no name attribute"
                )
        if weights is None:
            weights = [1.0] * len(objectives)
        weights = list(weights)
        if len(weights) != len(objectives):
            raise DataValidationError(
                f"{len(objectives)} objectives but {len(weights)} weight(s) "
                "-- the two lists must match"
            )
        for weight in weights:
            if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise DataValidationError(
                    f"weights must be finite numbers, got {weight!r}"
                )
            if weight <= 0:
                raise DataValidationError(
                    f"weights must be strictly positive, got {weight!r} "
                    "(a zero-weight objective should simply be removed)"
                )
        self.objectives = objectives
        self.weights = [float(weight) for weight in weights]

    @property
    def n_invalid(self) -> int:
        """Sum of the children's invalid-SMILES counters."""
        return int(
            sum(
                getattr(objective, "n_invalid", 0)
                for objective in self.objectives
            )
        )

    def reset_stats(self) -> None:
        """Zero the counters of the combinator AND its children.

        A ``reset_stats`` that left the children counting made the summed
        ``n_invalid`` outlive the reset (4 while ``n_scored`` went 0).
        Duck-typed children without the method are skipped, not crashed
        on.
        """
        super().reset_stats()
        for objective in self.objectives:
            reset = getattr(objective, "reset_stats", None)
            if callable(reset):
                reset()

    def score_with_breakdown(self, smiles: str) -> tuple[float, dict[str, float]]:
        """Return ``(total, {objective_name: score})``.

        Duplicate objective names are de-duplicated as ``name_2``,
        ``name_3`` ... so the breakdown stays addressable.  A child score
        that is NaN or infinite (a misbehaving custom objective) counts
        as ``0.0`` -- i.e. it vetoes the molecule, keeping the documented
        ``[0, 1]`` total contract true.
        """
        self._n_scored += 1
        used: set[str] = set()
        breakdown: dict[str, float] = {}
        for objective in self.objectives:
            base = str(getattr(objective, "name", "objective"))
            key, suffix = base, 1
            while key in used:
                suffix += 1
                key = f"{base}_{suffix}"
            used.add(key)
            try:
                value = float(objective.score(smiles))
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"objective {key!r} returned a non-numeric score for "
                    f"{smiles!r} -- objectives must map SMILES to floats"
                ) from exc
            if not math.isfinite(value):
                value = 0.0
            breakdown[key] = value
        if any(value <= 0.0 for value in breakdown.values()):
            return 0.0, breakdown
        # Weighted geometric mean in LOG space: the linear-space product
        # ``prod(v ** w)`` underflows to 0.0 long before the weights get
        # extreme (weights [2000, 2000] scored 0.5/0.5 as 0.0) and the
        # exponent sum overflows to inf ([1e308, 1e308] fabricated a
        # perfect 1.0 via ``0.0 ** 0.0``).  Only weight RATIOS matter, so
        # rescale by the max weight; math.fsum keeps the log-sum exact.
        max_weight = max(self.weights)
        log_total = math.fsum(
            math.log(value) * (weight / max_weight)
            for value, weight in zip(breakdown.values(), self.weights)
        )
        weight_sum = math.fsum(weight / max_weight for weight in self.weights)
        total = math.exp(log_total / weight_sum)
        return float(min(max(total, 0.0), 1.0)), breakdown

    def score(self, smiles: str) -> float:
        """Total weighted score (see :meth:`score_with_breakdown`)."""
        return self.score_with_breakdown(smiles)[0]

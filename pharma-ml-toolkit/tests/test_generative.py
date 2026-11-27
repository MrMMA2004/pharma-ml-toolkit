"""Tests for :mod:`pharmaml.generative` — the molecule-generation subpackage.

Covers, in order of the 0-to-100 story: the bundled corpus (validity,
canonicity, deduplication, size), the objective library (pinned numeric
spot-checks probed against RDKit 2026.09.1, band semantics, similarity,
substructure, multi-objective weighting + veto, invalid-SMILES
accounting, constructor validation), the SMILES genetic algorithm
(determinism, improvement, seeded mode, caller-input immutability,
history shape), the char-level RNN prior (torch lazy-import contract,
training, sampling validity/determinism, vocabulary safety), the
REINVENT-style RL optimizer (smoke, determinism, prior immutability,
hyperparameter guards), the ``pharmaml generative`` CLI twins, and the
result container.

Timings are tuned for the CPU sandbox: the shared module-scoped prior
fixture trains in <1 s and every optimiser test stays well under a second.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import random
import sys
from typing import ClassVar

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

# 100x-32-a: the generative package imports rdkit lazily (in-function, for
# the DependencyError contracts), so ``import rdkit`` alone leaves the
# ``rdkit.Chem`` attribute unset -- the 55 ``rdkit.Chem.*`` accesses below
# worked only when an earlier test happened to trigger the submodule load.
# Load the submodule once here and every attribute access is
# order-independent (fail-first: the isolated runs of
# TestScaffoldHopPathTransplants::test_linker_path_units_enumeration_is_
# pinned and ::test_excise_linker_path_builds_zippable_donor crashed with
# AttributeError: module 'rdkit' has no attribute 'Chem').
pytest.importorskip("rdkit.Chem")

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.generative import (
    DEFAULT_CORPUS,
    GenerationResult,
    GeneticOptimizer,
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    NoveltyObjective,
    ParetoFront,
    QedObjective,
    ReinventOptimizer,
    ScoringObjective,
    SimilarityObjective,
    SmilesRnn,
    SubstructureObjective,
    TpsaBandObjective,
    pareto_front,
    ring_topology_moves,
    ring_topology_mutate,
    run_genetic_optimization,
    run_rl_optimization,
    scaffold_hop_crossover,
)
from pharmaml.generative._utils import canonical_smiles
from pharmaml.generative.genetic import (
    _build_decoration_pool,
    _scaffold_mutate,
    _scaffold_split,
)


# (round-249) the module-level ``logging.disable(logging.CRITICAL)``
# this fixture replaces leaked for the WHOLE pytest process and silently
# broke ``caplog`` assertions in sibling modules sharing a pytest process
# (the nine-chunk gate caught it: test_format_converters before
# test_numpy_neural_checkpoints).  Same silence, correct scope: applied
# around each test of THIS module only, previous level restored after.
@pytest.fixture(autouse=True)
def _quiet_module_logging():
    prev = logging.root.manager.disable
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    prev_handlers = list(pharma_root.handlers)
    prev_level = pharma_root.level
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(prev)
    # (round-249b) this module's CLI tests call main() ->
    # configure_logging(), which flips pharmaml.propagate to False and
    # attaches a console handler bound to THIS test's captured stderr;
    # both would leak into sibling modules sharing the process (the
    # gate caught it via test_format_converters -> the §20.5 caplog
    # assertions).  Restore the at-import state after every test.
    pharma_root.propagate = prev_propagate
    pharma_root.handlers[:] = prev_handlers
    pharma_root.setLevel(prev_level)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
INVALID = "not_a_smiles"
#: Stereo-free cholesterol -- a steroid framework guaranteed structurally
#: distant from the bundled corpus (no steroids among the 244 entries;
#: probed max Morgan Tanimoto = 0.358 -> novelty 0.642 on rdkit 2026.09.1).
CHOLESTEROL = "CC(C)CCCC(C)C1CCC2C3CC=C4CCCC(O)CCC4(C)C3CCC12C"
#: Stereo-free squalene -- acyclic polyene, probed novelty 0.851.
SQUALENE = "CC(C)=CCCC(C)=CC(C)=CCCC(C)=CCCC(C)=CCC(C)C"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
NICOTINE = "CN1CCCC1c1cccnc1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hide_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a torch-less machine (import of torch raises ImportError)."""
    monkeypatch.setitem(sys.modules, "torch", None)


def _hide_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an RDKit-less machine (import of rdkit raises ImportError)."""
    monkeypatch.setitem(sys.modules, "rdkit", None)


def _n_valid(smiles_list: list[str]) -> int:
    """Number of RDKit-parseable strings in ``smiles_list``."""
    return sum(canonical_smiles(smiles) is not None for smiles in smiles_list)


@pytest.fixture(scope="module")
def trained_prior() -> SmilesRnn:
    """Shared char-GRU prior trained on the full corpus (8 epochs, <1 s)."""
    prior = SmilesRnn(hidden_dim=96, seed=42)
    prior.train_prior(DEFAULT_CORPUS, epochs=8)
    return prior


# ---------------------------------------------------------------------------
# The bundled corpus
# ---------------------------------------------------------------------------
class TestDefaultCorpus:
    def test_size_is_pinned_and_above_floor(self) -> None:
        # 244 curated entries (module docstring); the floor guards the
        # "diverse drug-like corpus" contract (~180 minimum).
        assert len(DEFAULT_CORPUS) == 244
        assert len(DEFAULT_CORPUS) >= 180

    def test_every_smiles_is_valid_and_canonical(self) -> None:
        from rdkit import Chem

        for smiles in DEFAULT_CORPUS:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"corpus entry does not parse: {smiles!r}"
            assert Chem.MolToSmiles(mol) == smiles, f"non-canonical: {smiles!r}"

    def test_corpus_is_deduplicated(self) -> None:
        assert len(set(DEFAULT_CORPUS)) == len(DEFAULT_CORPUS)

    def test_recognisable_drugs_are_present(self) -> None:
        for drug in (
            ASPIRIN,                       # aspirin
            PARACETAMOL,                   # paracetamol
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
            "CN(C)C(=N)N=C(N)N",           # metformin
            "CN1CCCC1c1cccnc1",            # nicotine
            "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
            "CN=C(NC#N)NCCSCc1[nH]cnc1C",              # cimetidine
        ):
            assert drug in DEFAULT_CORPUS, drug

    def test_corpus_is_an_immutable_tuple(self) -> None:
        # A module-level *list* would let one caller's append/sort corrupt
        # the default corpus for the whole process (100x-21-b fix).
        assert isinstance(DEFAULT_CORPUS, tuple)
        with pytest.raises(AttributeError):
            DEFAULT_CORPUS.append("CCO")  # type: ignore[union-attr]

    def test_smiles_lengths_are_bounded(self) -> None:
        lengths = [len(smiles) for smiles in DEFAULT_CORPUS]
        assert 3 <= min(lengths) and max(lengths) <= 60

    def test_named_drugs_match_published_weights(self) -> None:
        # 100x-21-b corpus review: 15 "named drug" entries were structurally
        # wrong (e.g. the "pregabalin" entry was leucine, "menthol" carried
        # +98 Da of extra skeleton, cimetidine/methimazole/trimethoprim had
        # the wrong heteroaromatic ring).  Every corrected structure is
        # pinned here by canonical SMILES AND by its published molecular
        # weight, so a future corpus edit cannot reintroduce a lookalike.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        published = {
            "cimetidine": ("CN=C(NC#N)NCCSCc1[nH]cnc1C", 252.346),
            "naproxen": ("COc1ccc2cc(C(C)C(=O)O)ccc2c1", 230.262),
            "salbutamol": ("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", 239.311),
            "lidocaine": ("CCN(CC)CC(=O)Nc1c(C)cccc1C", 234.337),
            "procainamide": ("CCN(CC)CCNC(=O)c1ccc(N)cc1", 235.325),
            "trimethoprim": ("COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC", 290.321),
            "clofibrate": ("CCOC(=O)C(C)(C)Oc1ccc(Cl)cc1", 242.699),
            "pregabalin": ("CC(C)CCC(CN)C(=O)O", 159.226),
            "gabapentin": ("NCC1(CC(=O)O)CCCCC1", 171.237),
            "amantadine": ("NC1C2CC3CC(C2)CC1C3", 151.248),
            "mefenamic acid": ("Cc1cccc(Nc2ccccc2C(=O)O)c1C", 241.289),
            "chlorpropamide": ("CCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1", 276.745),
            "methimazole": ("Cn1cc[nH]c1=S", 114.169),
            "menthol": ("CC(C)C1CCC(C)C(O)C1", 156.265),
            "thymol": ("Cc1ccc(O)c(C(C)C)c1", 150.221),
        }
        for name, (smiles, weight) in published.items():
            assert smiles in DEFAULT_CORPUS, f"{name} missing from the corpus"
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, name
            assert Descriptors.MolWt(mol) == pytest.approx(weight, abs=0.05), (
                f"{name} does not match its published molecular weight"
            )

    def test_amino_acids_and_nucleobases_are_the_real_ones(self) -> None:
        # 100x-21-b corpus review, part 2: four "amino acid" entries were
        # one-methylene homologs (homophenylalanine, S-methylcysteine,
        # ornithine, 4-guanidino-2-aminobutyric acid instead of Phe/Met/
        # Lys/Arg) and one "nucleobase" entry was a C6H5N5O imidazo-fused
        # pyridinone that is no purine/pyrimidine at all -- guanine was
        # missing entirely.  Every corrected entry is pinned by canonical
        # SMILES AND published molecular weight; the homologs are pinned
        # ABSENT so a future edit cannot silently regress to a lookalike.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        real = {
            "phenylalanine": ("NC(Cc1ccccc1)C(=O)O", 165.192),
            "methionine": ("CSCCC(N)C(=O)O", 149.215),
            "lysine": ("NCCCCC(N)C(=O)O", 146.190),
            "arginine": ("N=C(N)NCCCC(N)C(=O)O", 174.204),
            "guanine": ("Nc1nc2[nH]cnc2c(=O)[nH]1", 151.129),
        }
        for name, (smiles, weight) in real.items():
            assert smiles in DEFAULT_CORPUS, f"{name} missing from the corpus"
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, name
            assert Descriptors.MolWt(mol) == pytest.approx(weight, abs=0.05), (
                f"{name} does not match its published molecular weight"
            )
        for homolog in (
            "NC(CCc1ccccc1)C(=O)O",   # homophenylalanine
            "CSCC(N)C(=O)O",          # S-methylcysteine
            "NCCCC(N)C(=O)O",         # ornithine
            "NC(N)=NCCC(N)C(=O)O",    # 4-guanidino-2-aminobutyric acid
            "N=C1NC(=O)C2=NC=NC2=C1N",  # C6H5N5O non-nucleobase
        ):
            assert homolog not in DEFAULT_CORPUS, homolog

    def test_corpus_property_ranges_match_the_docstring(self) -> None:
        # The module docstring claims 46-365 Da, LogP -3.2 .. +4.7 and
        # 3-48 characters; pin the claims so a corpus edit that breaks
        # them (or a stale docstring) is caught.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mols = [Chem.MolFromSmiles(smiles) for smiles in DEFAULT_CORPUS]
        assert all(mol is not None for mol in mols)
        weights = [Descriptors.MolWt(mol) for mol in mols]
        logps = [Descriptors.MolLogP(mol) for mol in mols]
        lengths = [len(smiles) for smiles in DEFAULT_CORPUS]
        assert 45 <= min(weights) <= 47, min(weights)
        assert 364 <= max(weights) <= 366, max(weights)
        assert -3.3 <= min(logps) <= -3.1, min(logps)
        assert 4.6 <= max(logps) <= 4.8, max(logps)
        assert 3 <= min(lengths) and max(lengths) <= 48


# ---------------------------------------------------------------------------
# The objective library
# ---------------------------------------------------------------------------
class TestScoringObjectives:
    def test_qed_aspirin_is_pinned(self) -> None:
        # Probed against rdkit 2026.09.1: qed(aspirin) = 0.5501217966938848.
        assert QedObjective().score(ASPIRIN) == pytest.approx(0.550122, abs=1e-4)

    def test_invalid_smiles_scores_zero_with_accounting(self) -> None:
        objective = QedObjective()
        assert objective.score(INVALID) == 0.0
        assert objective.score("") == 0.0
        assert objective.n_invalid == 2
        assert objective.n_scored == 2
        objective.reset_stats()
        assert objective.n_invalid == 0 and objective.n_scored == 0

    def test_logp_band_scores_one_at_the_centre(self) -> None:
        # Aspirin's Crippen LogP is 1.3101 -> band [0.31, 2.31] centres it.
        objective = LogPBandObjective(0.31, 2.31)
        assert objective.score(ASPIRIN) == pytest.approx(1.0, abs=1e-6)

    def test_logp_band_decays_outside_the_window(self) -> None:
        # Metformin (LogP = -1.2438) is 2.55 sigma below the centre of
        # [0.31, 2.31]; exp(-0.5 * 2.55**2) ~ 0.039.  Ethanol (-0.0014)
        # sits 1.31 sigma out -> ~0.42.  Both probed, then pinned.
        objective = LogPBandObjective(0.31, 2.31)
        assert objective.score("CN(C)C(=N)N=C(N)N") == pytest.approx(0.039, abs=0.02)
        assert objective.score("CCO") == pytest.approx(0.424, abs=0.02)

    def test_band_edges_are_one_sigma(self) -> None:
        # A molecule exactly at a band edge is one sigma from the centre:
        # score = exp(-0.5) ~ 0.6065 (the documented +/-1 sigma semantics).
        import math

        from pharmaml.generative.scoring import _band_score

        assert _band_score(0.31, 0.31, 2.31) == pytest.approx(math.exp(-0.5))
        assert _band_score(2.31, 0.31, 2.31) == pytest.approx(math.exp(-0.5))
        assert _band_score(1.31, 0.31, 2.31) == pytest.approx(1.0)

    def test_mw_and_tpsa_band_objectives(self) -> None:
        # Aspirin: MolWt = 180.159, TPSA = 63.6 (probed).
        mw = MolecularWeightBandObjective(170, 190)
        assert mw.score(ASPIRIN) == pytest.approx(0.99987, abs=1e-4)
        tpsa = TpsaBandObjective(60, 70)
        assert tpsa.score(ASPIRIN) == pytest.approx(0.9616, abs=1e-3)

    def test_similarity_to_self_is_one(self) -> None:
        objective = SimilarityObjective(ASPIRIN)
        assert objective.score(ASPIRIN) == 1.0
        assert objective.score(objective.seed_smiles) == 1.0

    def test_similarity_aspirin_paracetamol_is_pinned(self) -> None:
        # Morgan radius-2, 2048-bit Tanimoto (probed: 2/9 shared bits).
        assert SimilarityObjective(ASPIRIN).score(PARACETAMOL) == pytest.approx(
            0.2222, abs=1e-3
        )

    def test_similarity_invalid_smiles_scores_zero(self) -> None:
        objective = SimilarityObjective(ASPIRIN)
        assert objective.score(INVALID) == 0.0
        assert objective.n_invalid == 1

    def test_substructure_objective(self) -> None:
        objective = SubstructureObjective("c1ccccc1")  # aromatic benzene ring
        assert objective.score(ASPIRIN) == 1.0
        assert objective.score("CCO") == 0.0
        # a real pharmacophore SMARTS: amide carbonyl
        amide = SubstructureObjective("[NX3][CX3](=[OX1])")
        assert amide.score(PARACETAMOL) == 1.0
        assert amide.score("CCO") == 0.0

    def test_multi_objective_weighted_geometric_mean(self) -> None:
        objective = MultiObjective(
            [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
        )
        total, breakdown = objective.score_with_breakdown(ASPIRIN)
        assert breakdown["qed"] == pytest.approx(0.550122, abs=1e-4)
        assert breakdown["logp_band"] == pytest.approx(0.788218, abs=1e-4)
        expected = 0.550122**0.7 * 0.788218**0.3  # weights sum to 1.0
        assert total == pytest.approx(expected, abs=1e-4)
        assert 0.0 <= total <= 1.0

    def test_multi_objective_veto_masks_nonzero_scores(self) -> None:
        # Ethanol has QED > 0 but no cyclopropane ring: the hard constraint
        # vetoes the whole molecule.
        objective = MultiObjective(
            [QedObjective(), SubstructureObjective("C1CC1")], weights=[1.0, 1.0]
        )
        total, breakdown = objective.score_with_breakdown("CCO")
        assert breakdown["qed"] > 0.0
        assert breakdown["substructure"] == 0.0
        assert total == 0.0

    def test_multi_objective_duplicate_names_are_disambiguated(self) -> None:
        objective = MultiObjective(
            [SubstructureObjective("c1ccccc1"), SubstructureObjective("C1CC1")]
        )
        _, breakdown = objective.score_with_breakdown(ASPIRIN)
        assert set(breakdown) == {"substructure", "substructure_2"}

    def test_custom_objective_subclass(self) -> None:
        class HeavyAtomObjective(ScoringObjective):
            name = "heavy_atoms"

            def score(self, smiles: str) -> float:
                mol = self._parse(smiles)
                if mol is None:
                    return 0.0
                return min(mol.GetNumHeavyAtoms() / 30.0, 1.0)

        objective = HeavyAtomObjective()
        assert objective.score(ASPIRIN) == pytest.approx(13 / 30)
        assert objective.score(INVALID) == 0.0
        assert objective.n_invalid == 1
        assert objective.score_many([ASPIRIN, "CCO"]) == [
            pytest.approx(13 / 30),
            pytest.approx(3 / 30),
        ]

    def test_band_constructor_errors(self) -> None:
        with pytest.raises(DataValidationError, match="greater than hi"):
            LogPBandObjective(3.0, 1.0)
        with pytest.raises(DataValidationError, match="finite number"):
            LogPBandObjective(float("nan"), 1.0)
        with pytest.raises(DataValidationError, match="positive"):
            MolecularWeightBandObjective(0, 100)
        with pytest.raises(DataValidationError, match="positive"):
            TpsaBandObjective(-5, 5)

    def test_similarity_and_substructure_constructor_errors(self) -> None:
        with pytest.raises(DataValidationError, match="parseable SMILES"):
            SimilarityObjective(INVALID)
        with pytest.raises(DataValidationError, match="non-empty string"):
            SimilarityObjective("")
        with pytest.raises(DataValidationError, match="negative int"):
            SimilarityObjective(ASPIRIN, radius=-1)
        with pytest.raises(DataValidationError, match="parseable SMARTS"):
            SubstructureObjective("???")
        with pytest.raises(DataValidationError, match="non-empty string"):
            SubstructureObjective(" ")

    def test_bool_arguments_are_rejected_like_novelty(self) -> None:
        # Regression (26-R2): bools ARE ints, so radius=True passed the
        # numeric guard (silently treated as radius 1) while
        # NoveltyObjective raised a typed error for the very same input;
        # LogPBandObjective(False, True) silently became the band [0, 1].
        # Novelty's exact bool-exclusion pattern now guards every numeric
        # parameter of the similarity and band objectives.
        with pytest.raises(DataValidationError, match="radius"):
            SimilarityObjective(ASPIRIN, radius=True)
        with pytest.raises(DataValidationError, match="radius"):
            SimilarityObjective(ASPIRIN, radius=False)
        with pytest.raises(DataValidationError, match="n_bits"):
            SimilarityObjective(ASPIRIN, n_bits=True)
        with pytest.raises(DataValidationError, match="finite number"):
            LogPBandObjective(False, True)  # used to be the band [0, 1]
        with pytest.raises(DataValidationError, match="finite number"):
            MolecularWeightBandObjective(True, 200.0)  # used to be [1, 200]
        with pytest.raises(DataValidationError, match="finite number"):
            TpsaBandObjective(True, 40.0)  # used to be [1, 40]
        # genuine ints keep working
        assert SimilarityObjective(ASPIRIN, radius=2, n_bits=2048).score(ASPIRIN) == 1.0
        assert LogPBandObjective(0, 2).score("CCO") > 0.0

    def test_multi_objective_constructor_errors(self) -> None:
        with pytest.raises(DataValidationError, match="at least two"):
            MultiObjective([QedObjective()], weights=[1.0])
        with pytest.raises(DataValidationError, match="must match"):
            MultiObjective([QedObjective(), LogPBandObjective(1, 3)], weights=[1.0])
        with pytest.raises(DataValidationError, match="strictly positive"):
            MultiObjective(
                [QedObjective(), LogPBandObjective(1, 3)], weights=[1.0, 0.0]
            )
        with pytest.raises(DataValidationError, match="callable score"):
            MultiObjective([QedObjective(), "not an objective"])

    def test_multi_objective_duck_typed_protocol(self) -> None:
        # The docstring promises "any object exposing name and a callable
        # score" -- not just ScoringObjective subclasses.
        class DuckObjective:
            name = "duck"

            def __init__(self, value: float) -> None:
                self.value = value

            def score(self, smiles: str) -> float:
                return self.value

        objective = MultiObjective([DuckObjective(0.5), DuckObjective(0.8)])
        total, breakdown = objective.score_with_breakdown("CCO")
        assert breakdown == {"duck": 0.5, "duck_2": 0.8}
        assert total == pytest.approx((0.5 * 0.8) ** 0.5)
        # duck-typed children without counters contribute zero to n_invalid
        assert objective.n_invalid == 0

    def test_multi_objective_non_finite_child_scores_veto(self) -> None:
        # Regression: a NaN-returning child made the total NaN (``nan <= 0``
        # is False, so the veto never fired) -- violating the documented
        # [0, 1] total contract.  Non-finite children now count as 0.0,
        # matching how the optimisers clamp misbehaving objectives.
        class NonFiniteObjective:
            def __init__(self, value: float) -> None:
                self.value = value
                self.name = f"child_{self.value}"

            def score(self, smiles: str) -> float:
                return self.value

        nan_obj = MultiObjective(
            [QedObjective(), NonFiniteObjective(float("nan"))]
        )
        total, breakdown = nan_obj.score_with_breakdown(ASPIRIN)
        assert breakdown["child_nan"] == 0.0
        assert total == 0.0

        inf_obj = MultiObjective(
            [QedObjective(), NonFiniteObjective(float("inf"))]
        )
        total, _ = inf_obj.score_with_breakdown(ASPIRIN)
        assert total == 0.0

        class NoneObjective:
            name = "none_child"

            def score(self, smiles: str) -> None:
                return None

        with pytest.raises(DataValidationError, match="non-numeric score"):
            MultiObjective([QedObjective(), NoneObjective()]).score(ASPIRIN)

    def test_multi_objective_reset_stats_resets_children(self) -> None:
        # Regression (26-R2): reset_stats() zeroed the combinator's own
        # counters but left the children counting -- n_invalid (a sum over
        # the children) stayed at 4 while n_scored went 0.
        multi = MultiObjective([QedObjective(), LogPBandObjective(1.0, 3.0)])
        for _ in range(2):
            multi.score(INVALID)
        assert multi.n_scored == 2
        assert multi.n_invalid == 4  # both children saw 2 invalid inputs
        multi.reset_stats()
        assert multi.n_scored == 0
        assert multi.n_invalid == 0
        assert [child.n_invalid for child in multi.objectives] == [0, 0]
        # duck-typed children without counters are skipped, not crashed on
        class DuckObjective:
            name = "duck"

            def score(self, smiles: str) -> float:
                return 0.5

        duck_multi = MultiObjective([QedObjective(), DuckObjective()])
        duck_multi.score(INVALID)
        assert duck_multi.n_invalid == 1  # only the counting child
        duck_multi.reset_stats()  # must not raise
        assert duck_multi.n_invalid == 0

    def test_multi_objective_extreme_weights_compute_in_log_space(self) -> None:
        # Regression (26-R2): the weighted geometric mean multiplied in
        # LINEAR space -- weights [2000, 2000] underflowed prod(v**w) to
        # 0.0 (children 0.5/0.5 -> 0.0!), [1e308, 1e308] overflowed the
        # exponent sum to inf and fabricated a perfect 1.0 (0.0 ** 0.0),
        # and mixed [1e307, 1e308] collapsed to 0.0 (expected ~0.767).
        # Log-space with math.fsum and max-rescaled weights (only ratios
        # matter) computes all three; v <= 0 still short-circuits to 0.
        class FixedObjective:
            def __init__(self, key: str, value: float) -> None:
                self.name = key
                self._value = value

            def score(self, smiles: str) -> float:
                return self._value

        def total(weights: list[float], values: list[float]) -> float:
            return MultiObjective(
                [FixedObjective("a", values[0]), FixedObjective("b", values[1])],
                weights=weights,
            ).score("CCO")

        assert total([2000.0, 2000.0], [0.5, 0.5]) == pytest.approx(0.5, rel=1e-12)
        assert total([1e308, 1e308], [0.5, 0.5]) == pytest.approx(0.5, rel=1e-12)
        # 0.5 ** (1/11) * 0.8 ** (10/11), probed: 0.7665378865
        assert total([1e307, 1e308], [0.5, 0.8]) == pytest.approx(
            0.5 ** (1 / 11) * 0.8 ** (10 / 11), rel=1e-9
        )
        # the zero-score veto survives the rewrite
        assert total([1.0, 1.0], [0.0, 0.8]) == 0.0

    def test_rdkit_absent_raises_dependency_error(self, monkeypatch) -> None:
        _hide_rdkit(monkeypatch)
        with pytest.raises(DependencyError, match="RDKit"):
            QedObjective().score("CCO")
        with pytest.raises(DependencyError, match=r"-e \'\.\[chem\]\'"):
            SimilarityObjective(ASPIRIN)

    def test_scoring_module_imports_without_rdkit(self, monkeypatch) -> None:
        _hide_rdkit(monkeypatch)
        spec = importlib.util.find_spec("pharmaml.generative.scoring")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "MultiObjective")


# ---------------------------------------------------------------------------
# The genetic algorithm
# ---------------------------------------------------------------------------
class TestGeneticOptimizer:
    def test_same_seed_reproduces_the_run(self) -> None:
        config = {"population_size": 32, "generations": 10, "seed": 42}
        first = GeneticOptimizer(QedObjective(), **config).optimize()
        second = GeneticOptimizer(QedObjective(), **config).optimize()
        assert first.smiles == second.smiles
        assert first.scores == second.scores
        assert first.history == second.history

    def test_different_seed_changes_the_run(self) -> None:
        first = GeneticOptimizer(
            QedObjective(), population_size=24, generations=4, seed=1
        ).optimize()
        second = GeneticOptimizer(
            QedObjective(), population_size=24, generations=4, seed=2
        ).optimize()
        assert first.smiles != second.smiles or first.history != second.history

    def test_final_best_beats_initial_population(self) -> None:
        # A far LogP target: the random initial population cannot contain
        # the optimum, so evolution has to improve on it.
        objective = LogPBandObjective(4.0, 6.0)
        result = GeneticOptimizer(
            objective, population_size=32, generations=15, seed=3
        ).optimize()
        initial = result.history[0]["best_score"]
        assert result.history[-1]["best_score"] > initial
        assert result.best_score == pytest.approx(result.history[-1]["best_score"])

    def test_best_score_is_monotone_across_generations(self) -> None:
        result = GeneticOptimizer(
            QedObjective(), population_size=24, generations=8, seed=9
        ).optimize()
        best_scores = [entry["best_score"] for entry in result.history]
        assert best_scores == sorted(best_scores)  # elitism guarantee

    def test_seeded_mode_anchors_the_population(self) -> None:
        objective = SimilarityObjective(ASPIRIN)
        result = GeneticOptimizer(
            objective, population_size=32, generations=8, seed=1
        ).optimize(seed_smiles=[ASPIRIN, PARACETAMOL])
        # generation 0 is scored before any evolution: the best of the
        # seed molecules leads it, and both seeds are kept in metadata.
        assert result.history[0]["best_smiles"] == ASPIRIN
        assert result.metadata["seed_molecules"] == [ASPIRIN, PARACETAMOL]
        assert ASPIRIN in result.smiles
        assert result.best_score >= 0.9  # aspirin itself scores 1.0

    def test_seeded_mode_does_not_mutate_caller_inputs(self) -> None:
        seeds = [ASPIRIN, PARACETAMOL]
        corpus = list(DEFAULT_CORPUS[:20])
        seeds_snapshot, corpus_snapshot = list(seeds), list(corpus)
        GeneticOptimizer(
            QedObjective(), population_size=16, generations=3, seed=5
        ).optimize(seed_smiles=seeds, corpus=corpus)
        assert seeds == seeds_snapshot
        assert corpus == corpus_snapshot

    def test_invalid_seeds_are_dropped_but_not_fatal(self) -> None:
        result = GeneticOptimizer(
            QedObjective(), population_size=16, generations=2, seed=5
        ).optimize(seed_smiles=["CCO", INVALID])
        assert result.metadata["seed_molecules"] == ["CCO"]

    def test_all_invalid_seeds_raise(self) -> None:
        with pytest.raises(DataValidationError, match="no valid seed"):
            GeneticOptimizer(
                QedObjective(), population_size=16, generations=2, seed=5
            ).optimize(seed_smiles=[INVALID, "also bad(("])

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(DataValidationError, match="corpus is empty"):
            GeneticOptimizer(
                QedObjective(), population_size=16, generations=2, seed=5
            ).optimize(corpus=[])

    def test_invalid_corpus_entries_warn_when_dropped(self, caplog) -> None:
        # Regression (26-R2): unparseable and non-string corpus entries
        # vanished SILENTLY (silent data loss) while invalid SEEDS always
        # warned -- the corpus must warn too, sibling-style.  20 valid
        # entries + 4 garbage: the warnings fire and the population draws
        # only from the surviving valid pool.
        import logging as _logging

        # the suite's two-silencer caplog pattern (cf.
        # TestNoveltyObjective.test_non_string_corpus_entries_warn_when_dropped)
        prev_disable = _logging.root.manager.disable
        pharma_root = _logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        _logging.disable(_logging.NOTSET)
        pharma_root.propagate = True
        try:
            with caplog.at_level(
                _logging.WARNING, logger="pharmaml.generative.genetic"
            ):
                result = GeneticOptimizer(
                    QedObjective(), population_size=16, generations=1, seed=1
                ).optimize(
                    corpus=[*DEFAULT_CORPUS[:20], INVALID, "(((((", "@@@", 123]
                )
        finally:
            pharma_root.propagate = prev_propagate
            _logging.disable(prev_disable)
        warnings_seen = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == _logging.WARNING
        ]
        assert any(
            "invalid" in message and "corpus" in message for message in warnings_seen
        ), warnings_seen
        assert any("non-string" in message for message in warnings_seen), warnings_seen
        # generation 0 is scored before any evolution, so every produced
        # molecule is a member of the valid pool
        valid_pool = set(DEFAULT_CORPUS[:20])
        assert result.smiles and set(result.smiles) <= valid_pool

    def test_tiny_corpus_is_duplicated_not_hung(self) -> None:
        # Regression: a corpus with fewer unique molecules than
        # population_size used to spin FOREVER in the unique-fill loop
        # (verified: timeout-20 probe of the pre-fix code never returned).
        # The documented contract is duplication with a warning.
        import random

        optimizer = GeneticOptimizer(
            QedObjective(), population_size=16, generations=2, seed=3
        )
        population, _ = optimizer._initial_population(
            None, ["CCO", "CCN", "CCOCC"], random.Random(3)
        )
        assert len(population) == 16
        assert set(population) <= {"CCO", "CCN", "CCOCC"}
        result = optimizer.optimize(corpus=["CCO", "CCN", "CCOCC"])
        assert len(result) > 0
        assert len(result.history) == 2

    def test_duplicate_canonical_spellings_are_deduplicated(self) -> None:
        # "OCC" canonicalizes to "CCO": the pool must deduplicate canonical
        # forms, or the unique-fill loop spins on duplicate spellings too
        # (pre-fix, optimize() hung forever on exactly this corpus).
        result = GeneticOptimizer(
            QedObjective(), population_size=12, generations=3, seed=3
        ).optimize(corpus=["CCO", "OCC", "CCS"])
        assert len(result.history) == 3
        # generation 0 holds the deduplicated pool: CCO + CCS, no raw "OCC"
        assert result.history[0]["n_unique"] == 2
        assert result.history[0]["best_smiles"] in ("CCO", "CCS")

    def test_non_string_corpus_entries_error_names_the_problem(self) -> None:
        # Regression: corpus=[None, 123] used to raise "corpus is empty",
        # hiding that entries existed but none were strings.
        with pytest.raises(DataValidationError, match="no string entries"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=1, seed=1
            ).optimize(corpus=[None, 123, b"CCO"])

    def test_bare_string_seed_smiles_is_a_typed_error(self) -> None:
        # Regression (26-R2): seed_smiles="CCO" silently iterated
        # CHARACTERS (['C', 'C', 'O'] single-atom seed molecules) -- the
        # exact trap NoveltyObjective guards against for its corpus.  The
        # same typed DataValidationError fires now.
        with pytest.raises(DataValidationError, match="sequence of SMILES"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=1, seed=1
            ).optimize(seed_smiles="CCO")

    def test_bare_string_corpus_is_a_typed_error(self) -> None:
        # Regression (26-R2): corpus="CCO" silently built a corpus of the
        # single atoms C, C, O -- the run's best molecule was literally
        # 'C'.  Same typed error as NoveltyObjective.
        with pytest.raises(DataValidationError, match="sequence of SMILES"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=1, seed=1
            ).optimize(corpus="CCO")

    def test_seeds_exceeding_population_are_all_scored(self) -> None:
        # Disclosed pinning test (behaviour documented in 100x-21-b): more
        # valid seeds than population_size are NEVER discarded -- generation
        # 0 scores them all, and the population contracts afterwards.
        seeds = ["CCO", "CCN", "CCS", "CCF", "CCCl", "CCBr", "CCI",
                 "CC=O", "CCC", "CCCO", "CCCCO", "CCCCCO"]
        result = GeneticOptimizer(
            QedObjective(), population_size=8, generations=3, seed=1
        ).optimize(seed_smiles=seeds, corpus=["CCO", "CCN"])
        assert result.history[0]["n_unique"] == 12
        assert result.history[1]["n_unique"] <= 8
        assert len(result.metadata["seed_molecules"]) == 12

    def test_history_shape_and_ranges(self) -> None:
        result = GeneticOptimizer(
            QedObjective(), population_size=24, generations=6, seed=42
        ).optimize()
        assert len(result.history) == 6
        for entry in result.history:
            assert set(entry) == {
                "generation",
                "best_score",
                "mean_score",
                "best_smiles",
                "n_unique",
                "valid_offspring_fraction",
            }
            assert 0.0 <= entry["best_score"] <= 1.0
            assert 0.0 <= entry["mean_score"] <= 1.0
            assert 0.0 <= entry["valid_offspring_fraction"] <= 1.0
            assert 0 < entry["n_unique"] <= 24
        assert [entry["generation"] for entry in result.history] == list(range(6))

    def test_result_ranking_and_metadata(self) -> None:
        result = GeneticOptimizer(
            QedObjective(), population_size=24, generations=4, seed=42
        ).optimize()
        assert result.scores == sorted(result.scores, reverse=True)
        assert result.best == result.smiles[0]
        assert result.best_score == result.scores[0]
        assert len(set(result.smiles)) == len(result.smiles)  # deduplicated
        assert result.metadata["method"] == "genetic_algorithm"
        assert result.metadata["seed"] == 42
        assert result.metadata["iterations"] == 4
        assert result.metadata["elapsed_s"] > 0
        assert result.metadata["valid_generated"] > 0

    def test_result_dataframe_and_summary(self) -> None:
        objective = MultiObjective(
            [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
        )
        result = GeneticOptimizer(
            objective, population_size=24, generations=4, seed=42
        ).optimize()
        frame = result.to_dataframe()
        assert list(frame.columns) == ["rank", "smiles", "score", "qed", "logp_band"]
        assert frame["rank"].tolist() == list(range(1, len(result) + 1))
        assert len(result) > 0
        assert "genetic_algorithm" in result.summary()
        assert "best" in result.summary()

    def test_hyperparameter_validation(self) -> None:
        for kwargs in (
            {"population_size": 2},
            {"generations": 0},
            {"mutation_rate": 0.0},
            {"mutation_rate": 1.5},
            {"crossover_rate": -0.1},
            {"elitism": 24, "population_size": 24},
            {"tournament_k": 1},
            {"max_smiles_length": 10},
        ):
            with pytest.raises(DataValidationError):
                GeneticOptimizer(QedObjective(), **kwargs)
        with pytest.raises(DataValidationError, match="score"):
            GeneticOptimizer(objective="qed?")

    def test_non_numeric_objective_return_is_a_typed_error(self) -> None:
        # Consistency with MultiObjective: a duck-typed objective returning
        # None used to escape as a raw TypeError from float(); the typed
        # error names the objective and the contract.
        class NoneScoreObjective:
            name = "none_score"

            def score(self, smiles: str):
                return None

        with pytest.raises(DataValidationError, match="non-numeric score"):
            GeneticOptimizer(
                NoneScoreObjective(), population_size=8, generations=1, seed=1
            ).optimize(corpus=["CCO", "CCN"])

    def test_run_genetic_optimization_convenience(self) -> None:
        result = run_genetic_optimization(
            QedObjective(), population_size=16, generations=3, seed=7
        )
        assert result.metadata["method"] == "genetic_algorithm"
        assert len(result) > 0


# ---------------------------------------------------------------------------
# The RNN prior
# ---------------------------------------------------------------------------
pytest.importorskip("torch", reason="the SMILES RNN needs the optional deep extra")


class TestSmilesRnn:
    def test_train_prior_returns_per_epoch_losses(self, trained_prior) -> None:
        losses = trained_prior.train_prior(DEFAULT_CORPUS, epochs=1)
        assert len(losses) == 1
        assert all(loss > 0 for loss in losses)

    def test_training_improves_the_loss(self) -> None:
        prior = SmilesRnn(hidden_dim=64, seed=0)
        losses = prior.train_prior(DEFAULT_CORPUS[:120], epochs=3)
        assert losses[-1] < losses[0]

    def test_is_trained_and_vocab_contains_specials(self, trained_prior) -> None:
        assert trained_prior.is_trained
        vocab = trained_prior.vocab
        assert vocab is not None
        for token in ("<pad>", "<unk>", "<start>", "<end>"):
            assert token in vocab
        corpus_chars = {char for smiles in DEFAULT_CORPUS for char in smiles}
        assert corpus_chars <= set(vocab)

    def test_sample_yields_valid_smiles(self, trained_prior) -> None:
        # Probed: 8/32 valid at temperature 1.0, seed 3 (an 8-epoch prior on
        # the full corpus).  The threshold keeps a >2x margin.
        sampled = trained_prior.sample(32, temperature=1.0, seed=3)
        assert len(sampled) == 32
        assert _n_valid(sampled) >= 3

    def test_sample_is_deterministic_given_seed(self, trained_prior) -> None:
        first = trained_prior.sample(16, temperature=0.9, seed=5)
        second = trained_prior.sample(16, temperature=0.9, seed=5)
        assert first == second
        other = trained_prior.sample(16, temperature=0.9, seed=6)
        assert other != first

    def test_sample_only_emits_vocabulary_characters(self, trained_prior) -> None:
        sampled = trained_prior.sample(16, temperature=1.2, seed=8)
        vocab_chars = set("".join(DEFAULT_CORPUS))
        assert all(set(smiles) <= vocab_chars for smiles in sampled)
        # special tokens are structural, never emitted
        assert all("<" not in smiles for smiles in sampled)

    def test_sample_respects_the_length_cap(self, trained_prior) -> None:
        sampled = trained_prior.sample(8, temperature=1.0, seed=2, max_length=25)
        assert all(len(smiles) <= 25 for smiles in sampled)

    def test_training_is_deterministic(self) -> None:
        config = {"embedding_dim": 48, "hidden_dim": 64, "seed": 11}
        first, second = SmilesRnn(**config), SmilesRnn(**config)
        losses_a = first.train_prior(DEFAULT_CORPUS[:80], epochs=2)
        losses_b = second.train_prior(DEFAULT_CORPUS[:80], epochs=2)
        assert losses_a == losses_b
        assert first.sample(8, seed=1) == second.sample(8, seed=1)

    def test_copy_is_independent_of_the_original(self, trained_prior) -> None:
        clone = trained_prior.copy()
        assert clone is not trained_prior
        assert clone.vocab == trained_prior.vocab
        assert clone.sample(8, seed=3) == trained_prior.sample(8, seed=3)

    def test_untrained_sample_raises(self) -> None:
        with pytest.raises(DataValidationError, match="untrained"):
            SmilesRnn().sample(4)

    def test_sample_argument_validation(self, trained_prior) -> None:
        for bad_temperature in (0, -1, -0.5):
            with pytest.raises(DataValidationError, match="temperature"):
                trained_prior.sample(4, temperature=bad_temperature)
        with pytest.raises(DataValidationError, match="n must be"):
            trained_prior.sample(0)
        with pytest.raises(DataValidationError, match="max_length"):
            trained_prior.sample(4, max_length=0)

    def test_constructor_validation(self) -> None:
        with pytest.raises(DataValidationError, match="CPU-only"):
            SmilesRnn(device="cuda")
        with pytest.raises(DataValidationError, match="too small"):
            SmilesRnn(hidden_dim=4)
        with pytest.raises(DataValidationError, match="n_layers"):
            SmilesRnn(n_layers=9)

    def test_seed_must_be_torch_encodable(self) -> None:
        # Regression: seeds beyond 2**64 - 1 escaped as a raw torch
        # "ValueError: Overflow when unpacking long long" from
        # torch.manual_seed -- only at train/sample time, deep inside the
        # run.  The constructor now rejects them with a typed error naming
        # the accepted interval (random.Random-based GA seeds stay legal
        # at any magnitude; this is the torch-side contract).
        with pytest.raises(DataValidationError, match="range torch can seed"):
            SmilesRnn(seed=2**64)
        with pytest.raises(DataValidationError, match="range torch can seed"):
            SmilesRnn(seed=-(2**63) - 1)
        with pytest.raises(DataValidationError, match="must be an int"):
            SmilesRnn(seed="42")
        # the closed interval itself is accepted
        SmilesRnn(seed=2**64 - 1)
        SmilesRnn(seed=-(2**63))

    def test_rl_seed_must_be_torch_encodable(self) -> None:
        # the same guard on the RL optimizer (its master seed threads into
        # SmilesRnn and the per-iteration sampling generators)
        with pytest.raises(DataValidationError, match="range torch can seed"):
            ReinventOptimizer(QedObjective(), seed=2**64)
        ReinventOptimizer(QedObjective(), seed=2**64 - 1)

    def test_sample_seed_must_be_torch_encodable(self, trained_prior) -> None:
        with pytest.raises(DataValidationError, match="range torch can seed"):
            trained_prior.sample(2, seed=2**64)
        # boundary seeds sample fine
        assert len(trained_prior.sample(2, seed=2**63)) == 2

    def test_boundary_seed_derived_generators_wrap(self) -> None:
        # Regression (26-R2): the shuffle generator (seed + 1) and the
        # persistent sampler (seed + 2) escaped the validated interval at
        # its upper edge -- SmilesRnn(seed=2**64 - 1).train_prior(...)
        # crashed with torch's RAW "ValueError: Overflow when unpacking
        # long long".  Derived seeds wrap back into the interval; sums that
        # stay inside keep their exact historic value (determinism).
        from pharmaml.generative.rnn import _derived_seed

        # historic in-range sums keep their exact value
        assert _derived_seed(41, 1) == 42
        assert _derived_seed(40, 2) == 42
        assert _derived_seed(-(2**63), 1) == -(2**63) + 1
        # escaping sums wrap back into the torch-encodable interval
        assert _derived_seed(2**64 - 1, 1) == -(2**63)
        assert _derived_seed(2**64 - 1, 2) == -(2**63) + 1
        for derived in (_derived_seed(2**64 - 1, 1), _derived_seed(2**64 - 1, 2)):
            assert -(2**63) <= derived <= 2**64 - 1

        # end-to-end at the upper boundary: train + sample on BOTH the
        # persistent-generator (seed + 2) and explicit-seed paths
        first = SmilesRnn(hidden_dim=32, seed=2**64 - 1)
        losses = first.train_prior(["CCO", "CCN"], epochs=1)
        assert len(losses) == 1 and losses[0] > 0
        sampled = first.sample(4, seed=None)
        assert len(sampled) == 4
        assert len(first.sample(2, seed=2**64 - 1)) == 2
        # determinism on the wrapped path
        second = SmilesRnn(hidden_dim=32, seed=2**64 - 1)
        second.train_prior(["CCO", "CCN"], epochs=1)
        assert second.sample(4, seed=None) == sampled

    def test_train_prior_argument_validation(self, trained_prior) -> None:
        with pytest.raises(DataValidationError, match="empty"):
            trained_prior.train_prior([])
        with pytest.raises(DataValidationError, match="epochs"):
            trained_prior.train_prior(["CCO"], epochs=0)
        with pytest.raises(DataValidationError, match="batch_size"):
            trained_prior.train_prior(["CCO"], batch_size=0)

    def test_train_prior_lr_and_grad_clip_are_validated(self) -> None:
        # Regression: lr=0 trained silently (no learning at all), lr=-1/nan
        # escaped as torch's raw "ValueError: Invalid learning rate", and
        # lr=inf silently NaN-ed every weight (the next sample() then died
        # with the baffling "probability tensor contains inf/nan").
        # grad_clip=0 silently froze training the same way.  All are typed
        # errors now; a FRESH untrained RNN keeps the shared fixture safe.
        for bad_lr in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(DataValidationError, match="lr"):
                SmilesRnn().train_prior(["CCO", "CCN"], epochs=1, lr=bad_lr)
        for bad_clip in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(DataValidationError, match="grad_clip"):
                SmilesRnn().train_prior(["CCO", "CCN"], epochs=1, grad_clip=bad_clip)

    def test_non_finite_temperature_is_rejected(self, trained_prior) -> None:
        # Regression: temperature=nan passed the ``temperature <= 0`` guard
        # (NaN compares False) and crashed deep inside torch with the raw
        # "RuntimeError: probability tensor contains either inf, nan or
        # element < 0"; inf silently produced uniform noise.  Both are
        # typed errors now (RL already rejected them via its (0, 5] range).
        with pytest.raises(DataValidationError, match="temperature"):
            trained_prior.sample(4, temperature=float("nan"), seed=1)
        with pytest.raises(DataValidationError, match="temperature"):
            trained_prior.sample(4, temperature=float("inf"), seed=1)
        # finite positives still work, including the extremes
        assert len(trained_prior.sample(2, temperature=1e-6, seed=1)) == 2
        assert len(trained_prior.sample(2, temperature=100.0, seed=1)) == 2


class TestTorchAbsentContract:
    """The neural.py lazy-import contract, mirrored for the generative RNN."""

    def test_rnn_module_imports_without_torch(self, monkeypatch) -> None:
        _hide_torch(monkeypatch)
        spec = importlib.util.find_spec("pharmaml.generative.rnn")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "SmilesRnn")

    def test_rl_module_imports_without_torch(self, monkeypatch) -> None:
        _hide_torch(monkeypatch)
        spec = importlib.util.find_spec("pharmaml.generative.rl")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "ReinventOptimizer")

    def test_generative_package_imports_without_torch(self, monkeypatch) -> None:
        _hide_torch(monkeypatch)
        spec = importlib.util.find_spec("pharmaml.generative")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert "GeneticOptimizer" in module.__all__

    def test_train_prior_raises_dependency_error_without_torch(
        self, monkeypatch
    ) -> None:
        _hide_torch(monkeypatch)
        with pytest.raises(DependencyError, match="torch"):
            SmilesRnn().train_prior(["CCO"])
        with pytest.raises(DependencyError, match=r"-e '\.\[deep\]'"):
            SmilesRnn().train_prior(["CCO"])

    def test_sample_raises_dependency_error_without_torch(self, monkeypatch) -> None:
        _hide_torch(monkeypatch)
        prior = SmilesRnn()
        # a trained model object is required to reach the torch gate
        prior._vocab = ["<pad>", "<unk>", "<start>", "<end>", "C"]
        prior._model = object()
        with pytest.raises(DependencyError, match="torch"):
            prior.sample(2, seed=1)

    def test_rl_optimize_raises_dependency_error_without_torch(
        self, monkeypatch
    ) -> None:
        _hide_torch(monkeypatch)
        with pytest.raises(DependencyError, match="torch"):
            ReinventOptimizer(QedObjective(), iterations=1).optimize()
        with pytest.raises(DependencyError, match=r"-e '\.\[deep\]'"):
            ReinventOptimizer(QedObjective(), iterations=1).optimize()


# ---------------------------------------------------------------------------
# The RL optimizer
# ---------------------------------------------------------------------------
class TestReinventOptimizer:
    def test_smoke_run_produces_ranked_molecules(self, trained_prior) -> None:
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=4, batch_size=16, seed=7
        ).optimize()
        assert len(result) > 0
        assert all(0.0 <= score <= 1.0 for score in result.scores)
        assert all(math.isfinite(score) for score in result.scores)
        assert result.scores == sorted(result.scores, reverse=True)
        assert result.metadata["method"] == "reinforce_rl"
        assert result.metadata["iterations"] == 4
        assert result.metadata["prior"] == "user prior"
        assert isinstance(result.metadata["early_stopped"], bool)
        assert result.metadata["valid_generated"] > 0

    def test_rl_improves_the_mean_score(self, trained_prior) -> None:
        # 8 probed iterations: mean QED climbs from 0.128 to ~0.24-0.33.
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=8, batch_size=32, seed=42
        ).optimize()
        means = [entry["mean_score"] for entry in result.history]
        assert sum(means[-3:]) / 3 > means[0]

    def test_rl_improves_mean_over_a_tiny_corpus(self) -> None:
        # Sanity requirement of the 26-R2 baseline fix: with the running
        # baseline actually live (no per-batch mean cancellation), a QED
        # run over a tiny corpus must still IMPROVE the objective (probed:
        # mean climbs 0.000 -> ~0.15, best reaches ~0.51 in 12 iterations).
        result = ReinventOptimizer(
            QedObjective(), corpus=list(DEFAULT_CORPUS[:24]), prior_epochs=12,
            iterations=12, batch_size=16, seed=42,
        ).optimize()
        means = [entry["mean_score"] for entry in result.history]
        assert sum(means[-3:]) / 3 > means[0]

    def test_injected_baseline_changes_the_run(self, trained_prior, monkeypatch) -> None:
        # Regression (26-R2): the advertised "REINFORCE with a running
        # baseline" was mathematically inert -- per-batch standardisation
        # ((a - mean(a)) / std(a)) cancels ANY baseline exactly, so no
        # baseline value could ever influence a run.  The rl._BASELINE_INIT
        # seam proves the fix: injecting 0.5 vs the None default (seeded
        # from the first batch's mean) must produce different losses and
        # histories.
        from pharmaml.generative import rl as rl_module

        config = {"prior": trained_prior, "iterations": 3, "batch_size": 16, "seed": 7}
        monkeypatch.setattr(rl_module, "_BASELINE_INIT", None)
        default_run = ReinventOptimizer(QedObjective(), **config).optimize()
        monkeypatch.setattr(rl_module, "_BASELINE_INIT", 0.5)
        injected_run = ReinventOptimizer(QedObjective(), **config).optimize()

        assert [entry["loss"] for entry in default_run.history] != [
            entry["loss"] for entry in injected_run.history
        ]
        assert [entry["mean_score"] for entry in default_run.history] != [
            entry["mean_score"] for entry in injected_run.history
        ]
        # both runs stay healthy: finite losses, in-range scores
        for run in (default_run, injected_run):
            assert all(math.isfinite(entry["loss"]) for entry in run.history)
            assert all(0.0 <= score <= 1.0 for score in run.scores)

    def test_rl_is_deterministic_given_seed(self, trained_prior) -> None:
        config = {"prior": trained_prior, "iterations": 3, "batch_size": 16, "seed": 7}
        first = ReinventOptimizer(QedObjective(), **config).optimize()
        second = ReinventOptimizer(QedObjective(), **config).optimize()
        assert first.smiles == second.smiles
        assert first.scores == second.scores
        assert [h["mean_score"] for h in first.history] == [
            h["mean_score"] for h in second.history
        ]

    def test_rl_does_not_mutate_the_prior(self, trained_prior) -> None:
        before = trained_prior.sample(4, seed=99)
        ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=2, batch_size=8, seed=3
        ).optimize()
        assert trained_prior.sample(4, seed=99) == before

    def test_rl_history_shape(self, trained_prior) -> None:
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=3, batch_size=16, seed=7
        ).optimize()
        assert len(result.history) == 3
        for entry in result.history:
            assert set(entry) == {
                "iteration",
                "best_score",
                "mean_score",
                "best_smiles",
                "valid_fraction",
                "replay_size",
                "loss",
                # prior-drift diagnostics (100x-66)
                "drift",
                "entropy",
                "agent_logprob",
                "prior_logprob",
            }
            assert 0.0 <= entry["valid_fraction"] <= 1.0
            assert 0.0 <= entry["best_score"] <= 1.0
        assert [entry["iteration"] for entry in result.history] == [0, 1, 2]

    def test_untrained_prior_raises(self) -> None:
        with pytest.raises(DataValidationError, match="untrained"):
            ReinventOptimizer(QedObjective(), prior=SmilesRnn()).optimize()

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(DataValidationError, match="corpus is empty"):
            ReinventOptimizer(
                QedObjective(), corpus=[], iterations=1, batch_size=8
            ).optimize()

    def test_bare_string_corpus_is_a_typed_error(self) -> None:
        # Regression (26-R2): corpus="CCO" silently trained the prior on
        # the single atoms C, C, O -- the run's best molecule was 'C'.
        # The same typed DataValidationError NoveltyObjective raises for
        # bare strings guards the RL corpus at construction time.
        with pytest.raises(DataValidationError, match="sequence of SMILES"):
            ReinventOptimizer(QedObjective(), corpus="CCO", iterations=1)

    def test_boundary_master_seed_runs_end_to_end(self) -> None:
        # Regression (26-R2): seed=2**64 - 1 crashed the WHOLE RL run with
        # torch's raw overflow (the on-the-fly prior's shuffle generator is
        # seeded with seed + 1 inside train_prior).  Derived seeds wrap
        # now, so both interval boundaries run end-to-end.
        for seed in (2**64 - 1, -(2**63)):
            result = ReinventOptimizer(
                QedObjective(), corpus=["CCO", "CCN", "CCS"],
                iterations=2, batch_size=8, prior_epochs=1, seed=seed,
            ).optimize()
            assert len(result) > 0
            assert result.metadata["seed"] == seed

    def test_corpus_is_ignored_when_a_prior_is_given(self, trained_prior) -> None:
        # Documented contract: "corpus ... ignored when prior is given" --
        # even an empty one must not raise (it did before the guard moved
        # into the train-a-prior branch).
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, corpus=[], iterations=2,
            batch_size=8, seed=3,
        ).optimize()
        assert result.metadata["prior"] == "user prior"
        assert len(result) > 0

    def test_unparseable_corpus_entries_are_dropped(self) -> None:
        # The prior trains on the corpus's chemistry, not its typos: invalid
        # SMILES are dropped (with a warning) before train_prior.
        result = run_rl_optimization(
            QedObjective(), corpus=[INVALID, "CCO", "CCN"],
            prior_epochs=2, iterations=2, batch_size=8, seed=3,
        )
        assert result.metadata["prior"] == "corpus prior (2 molecules, 2 epochs)"
        assert len(result) > 0

    def test_hyperparameter_validation(self) -> None:
        for kwargs in (
            {"iterations": 0},
            {"batch_size": 2},
            {"temperature": 0.0},
            {"temperature": 6.0},
            {"prior_epochs": 0},
            {"lr": 0.0},
            {"anchor_weight": -1.0},
            {"entropy_bonus": -0.5},
            {"replay_size": -1},
            {"patience": 0},
            {"grad_clip": 0.0},
            {"max_smiles_length": 5},
        ):
            with pytest.raises(DataValidationError):
                ReinventOptimizer(QedObjective(), **kwargs)
        with pytest.raises(DataValidationError, match="score"):
            ReinventOptimizer(objective=None, iterations=1)

    def test_non_finite_hyperparameters_are_rejected(self) -> None:
        # Regression: the sign-only guards (``if x <= 0`` / ``if x < 0``)
        # let NaN through (NaN compares False to everything) and inf past
        # the upper-unbounded ones.  Empirically: lr=inf NaN-ed the agent
        # in ONE SGD step and the NEXT iteration's multinomial crashed
        # with the raw "probability tensor contains inf/nan"; anchor_weight
        # = nan silently DISABLED the mode-collapse anchor; tol=nan made
        # every iteration look non-improving.  All float hyperparameters
        # must be finite at construction time.
        for kwargs in (
            {"lr": float("nan")},
            {"lr": float("inf")},
            {"anchor_weight": float("nan")},
            {"anchor_weight": float("inf")},
            {"entropy_bonus": float("nan")},
            {"entropy_bonus": float("inf")},
            {"tol": float("nan")},
            {"grad_clip": float("nan")},
        ):
            with pytest.raises(DataValidationError):
                ReinventOptimizer(QedObjective(), iterations=1, **kwargs)

    def test_non_numeric_objective_return_is_a_typed_error(self, trained_prior) -> None:
        # Consistency with MultiObjective and the GA: a duck-typed objective
        # returning None used to escape as a raw TypeError from float().
        class NoneScoreObjective:
            name = "none_score"

            def score(self, smiles: str):
                return None

        with pytest.raises(DataValidationError, match="non-numeric score"):
            ReinventOptimizer(
                NoneScoreObjective(), prior=trained_prior, iterations=1,
                batch_size=8, seed=1,
            ).optimize()

    def test_run_rl_optimization_trains_a_corpus_prior(self) -> None:
        # Convenience path: no prior given -> one is trained on the corpus.
        result = run_rl_optimization(
            QedObjective(), prior_epochs=4, iterations=2, batch_size=16, seed=11
        )
        assert result.metadata["method"] == "reinforce_rl"
        assert "corpus prior" in result.metadata["prior"]
        assert len(result) > 0
        assert result.metadata["iterations"] == 2


# ---------------------------------------------------------------------------
# Prior-drift introspection (100x-66): the diagnostics the loop always
# computed for the anchor loss, now recorded in history instead of
# discarded -- the data behind plot_rl_introspection.
# ---------------------------------------------------------------------------
class TestRlIntrospection:
    def test_history_records_drift_entropy_and_logprobs(
        self, trained_prior
    ) -> None:
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=4, batch_size=16, seed=7
        ).optimize()
        for entry in result.history:
            assert math.isfinite(entry["drift"])
            assert entry["drift"] >= 0.0  # a squared quantity
            assert math.isfinite(entry["entropy"])
            assert entry["entropy"] >= 0.0
            assert math.isfinite(entry["agent_logprob"])
            assert math.isfinite(entry["prior_logprob"])

    def test_iteration_zero_drift_is_exactly_zero(self, trained_prior) -> None:
        # The agent starts as a bit-exact copy of the frozen prior and
        # takes its first gradient step only AFTER the diagnostics are
        # computed: iteration 0's drift is the float-exact 0.0 (the
        # figure's anchor point -- plot_rl_introspection's panel 3).
        result = ReinventOptimizer(
            QedObjective(), prior=trained_prior, iterations=1, batch_size=16, seed=7
        ).optimize()
        assert result.history[0]["drift"] == 0.0
        assert (
            result.history[0]["agent_logprob"]
            == result.history[0]["prior_logprob"]
        )

    def test_anchor_weight_zero_still_records_drift(self, trained_prior) -> None:
        # The diagnostic is independent of the penalty: anchor_weight=0
        # (vanilla REINFORCE-with-baseline) still records the drift it is
        # no longer paying for -- introspection, not just accounting.
        result = ReinventOptimizer(
            QedObjective(),
            prior=trained_prior,
            iterations=2,
            batch_size=16,
            seed=7,
            anchor_weight=0.0,
        ).optimize()
        assert result.metadata["anchor_weight"] == 0.0
        assert result.history[0]["drift"] == 0.0
        assert result.history[-1]["drift"] > 0.0  # the policy HAS moved

    def test_heavy_anchor_brakes_the_drift(self, trained_prior) -> None:
        # The mode-collapse brake, made measurable: with the same seed and
        # iterations, a heavy anchor (1.0) leaves the policy orders of
        # magnitude closer to the prior than vanilla REINFORCE (probed:
        # 0.737 vs 38.800 at seed 7 -- the inversion seen at anchor=0.3
        # disappears once the brake dominates the objective).
        config = {"prior": trained_prior, "iterations": 8, "batch_size": 16, "seed": 7}
        unanchored = ReinventOptimizer(
            QedObjective(), anchor_weight=0.0, **config
        ).optimize()
        anchored = ReinventOptimizer(
            QedObjective(), anchor_weight=1.0, **config
        ).optimize()
        assert (
            anchored.history[-1]["drift"] < unanchored.history[-1]["drift"]
        )


# ---------------------------------------------------------------------------
# The result container
# ---------------------------------------------------------------------------
class TestGenerationResult:
    def test_empty_result_defaults(self) -> None:
        result = GenerationResult()
        assert result.is_empty()
        assert len(result) == 0
        assert result.best == ""
        assert result.best_score == 0.0
        assert result.to_dataframe().empty

    def test_best_defaults_to_first_ranked_molecule(self) -> None:
        result = GenerationResult(
            smiles=["CCO", "CCN"], scores=[0.9, 0.4], best="", best_score=0.0
        )
        assert result.best == "CCO"
        assert result.best_score == 0.9
        assert not result.is_empty()

    def test_breakdown_columns_widen_the_dataframe(self) -> None:
        result = GenerationResult(
            smiles=["CCO", "CCN"],
            scores=[0.9, 0.4],
            breakdown=[{"qed": 0.9, "logp": 0.2}, {"qed": 0.4}],
        )
        frame = result.to_dataframe()
        assert list(frame.columns) == ["rank", "smiles", "score", "qed", "logp"]
        assert frame["logp"].isna().tolist() == [False, True]

    def test_breakdown_keys_colliding_with_reserved_columns_raise(self) -> None:
        # Regression (26-R2): a breakdown key literally named "score" (or
        # "rank" / "smiles") silently CLOBBERED the leaderboard column in
        # to_dataframe() -- the total-score column showed the breakdown
        # value instead.  Reserved collisions are typed errors at
        # construction now, with a rename suggestion.
        for reserved in ("rank", "smiles", "score"):
            with pytest.raises(DataValidationError, match="reserved"):
                GenerationResult(
                    smiles=["CCO"], scores=[0.5], breakdown=[{reserved: 0.123}]
                )
        # non-colliding keys keep working, and the score column keeps
        # showing the TOTAL score
        result = GenerationResult(
            smiles=["CCO"], scores=[0.5], breakdown=[{"qed": 0.5}]
        )
        assert result.to_dataframe()["score"].tolist() == [0.5]

    def test_empty_breakdown_rows_are_independent_dicts(self) -> None:
        # Regression: the default used a shared ``[{}] * n`` list, so
        # mutating one row's breakdown silently edited every row.
        result = GenerationResult(smiles=["CCO", "CCN"], scores=[0.9, 0.4])
        result.breakdown[0]["qed"] = 1.0
        assert result.breakdown[1] == {}
        assert len(result.breakdown) == 2

    def test_mismatched_lengths_raise(self) -> None:
        # Regression: mismatched smiles/scores used to either crash the
        # constructor with a bare IndexError (scores shorter than smiles)
        # or construct silently and explode later inside pandas with
        # "All arrays must be of the same length".
        with pytest.raises(DataValidationError, match="misaligned"):
            GenerationResult(smiles=["CCO", "CCN", "CCC"], scores=[0.5])
        with pytest.raises(DataValidationError, match="misaligned"):
            GenerationResult(smiles=["CCO"], scores=[0.5, 0.9])
        with pytest.raises(DataValidationError, match="misaligned"):
            GenerationResult(smiles=["CCO"], scores=[])
        with pytest.raises(DataValidationError, match="misaligned"):
            GenerationResult(smiles=["CCO", "CCN"], scores=[0.5, 0.6],
                             breakdown=[{"qed": 0.5}])
        with pytest.raises(DataValidationError, match="misaligned"):
            GenerationResult(smiles=["CCO"], scores=[0.5],
                             breakdown=[{"qed": 0.5}, {"qed": 0.6}])
        # an empty-breakdown list is still auto-filled per row (documented)
        result = GenerationResult(smiles=["CCO", "CCN"], scores=[0.5, 0.6],
                                  breakdown=[])
        assert result.breakdown == [{}, {}]

    def test_summary_mentions_method_and_best(self) -> None:
        result = GenerationResult(
            smiles=["CCO"], scores=[0.5], metadata={"method": "ga", "iterations": 3}
        )
        assert "ga" in result.summary()
        assert "CCO" in result.summary()


# ---------------------------------------------------------------------------
# The CLI twins
# ---------------------------------------------------------------------------
class TestGenerativeCommands:
    def test_ga_run_happy_path_writes_leaderboard(self, tmp_path, capsys) -> None:
        from pharmaml.cli import main

        out = tmp_path / "ga_qed.csv"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed",
            "--iterations", "6", "--population", "24", "--seed", "42",
            "--top", "3", "--out", str(out),
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "genetic_algorithm" in output
        assert "method   : ga" in output
        assert out.is_file()
        frame = pd.read_csv(out)
        assert list(frame.columns) == ["rank", "smiles", "score"]
        assert len(frame) > 0

    def test_rl_run_tiny_budget(self, capsys) -> None:
        from pharmaml.cli import main

        code = main([
            "generative", "run",
            "--method", "rl", "--recipe", "qed",
            "--iterations", "2", "--batch-size", "16",
            "--prior-epochs", "8", "--seed", "42", "--top", "3",
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "reinforce_rl" in output
        assert "method   : rl" in output

    def test_ga_scaffold_hop_flag_reports_mode(self, tmp_path, capsys) -> None:
        from pharmaml.cli import main

        out = tmp_path / "ga_hop.csv"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed",
            "--iterations", "6", "--population", "24", "--seed", "42",
            "--scaffold-hop", "--top", "3", "--out", str(out),
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "hop      : on" in output  # BREED crossover summary line
        assert out.is_file()
        frame = pd.read_csv(out)
        assert list(frame.columns) == ["rank", "smiles", "score"]
        assert len(frame) > 0

    def test_scaffold_hop_with_rl_method_is_a_typed_error(self, capsys) -> None:
        # --scaffold-hop is a GA crossover: asking an RL run for it must be
        # an actionable typed error, not a silently ignored flag.
        from pharmaml.cli import main

        code = main([
            "generative", "run",
            "--method", "rl", "--recipe", "qed",
            "--scaffold-hop", "--iterations", "1",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "--scaffold-hop" in error and "ga only" in error

    def test_scaffold_plus_scaffold_hop_is_the_structural_mode(self, capsys) -> None:
        # both structural flags: framework recombination + decoration resampling
        from pharmaml.cli import main

        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed",
            "--iterations", "5", "--population", "16", "--seed", "42",
            "--scaffold", "--scaffold-hop", "--top", "3",
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "scaffold : on" in output
        assert "hop      : on" in output

    def test_seeded_similar_recipe_from_csv(self, tmp_path, capsys) -> None:
        from pharmaml.cli import main

        seeds = tmp_path / "seeds.csv"
        pd.DataFrame(
            {"name": ["aspirin", "paracetamol"], "smiles": [ASPIRIN, PARACETAMOL]}
        ).to_csv(seeds, index=False)
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "similar",
            "--seeds", str(seeds),
            "--iterations", "5", "--population", "16", "--top", "3",
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "seeds    : 2 molecule(s)" in output
        assert "similarity" in output

    def test_seeds_table_without_smiles_column_is_a_typed_error(
        self, tmp_path, capsys
    ) -> None:
        # Regression: a table with no SMILES column silently fell back to
        # the FIRST column, so a name/ic50 table surfaced later as the
        # baffling "seed_smiles 'aspirin' is not a parseable SMILES".
        from pharmaml.cli import main

        seeds = tmp_path / "no_smiles.csv"
        pd.DataFrame(
            {"name": ["aspirin", "ibuprofen"], "ic50": [3.2, 12.5]}
        ).to_csv(seeds, index=False)
        code = main([
            "generative", "run", "--recipe", "similar", "--seeds", str(seeds),
            "--iterations", "1",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "no SMILES column found" in error
        assert "ic50" in error and "name" in error

    def test_seeds_table_column_alias_is_detected(self, tmp_path, capsys) -> None:
        # canonical_smiles-style headers are auto-detected (the docking
        # ligand-loader candidate list); the name column is ignored.
        from pharmaml.cli import main

        seeds = tmp_path / "aliases.csv"
        pd.DataFrame({
            "name": ["aspirin", "ethanol"],
            "canonical_smiles": [ASPIRIN, "CCO"],
        }).to_csv(seeds, index=False)
        code = main([
            "generative", "run", "--recipe", "similar", "--seeds", str(seeds),
            "--iterations", "2", "--population", "8", "--top", "2",
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "seeds    : 2 molecule(s)" in output
        assert "similarity" in output

    def test_similar_recipe_without_seeds_is_an_actionable_error(
        self, capsys
    ) -> None:
        from pharmaml.cli import main

        code = main(["generative", "run", "--recipe", "similar"])
        assert code == 1
        error = capsys.readouterr().err
        assert "error:" in error
        assert "similar" in error and "--seed-smiles" in error

    def test_bad_recipe_is_a_clean_argparse_error(self) -> None:
        from pharmaml.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main(["generative", "run", "--recipe", "bogus"])
        assert excinfo.value.code == 2

    def test_corpus_command_prints_the_bundled_corpus(self, tmp_path, capsys) -> None:
        from pharmaml.cli import main

        code = main(["generative", "corpus"])
        assert code == 0
        output = capsys.readouterr().out
        assert str(len(DEFAULT_CORPUS)) in output
        assert ASPIRIN in output

        out = tmp_path / "corpus.csv"
        code = main(["generative", "corpus", "--out", str(out)])
        assert code == 0
        frame = pd.read_csv(out)
        assert list(frame.columns) == ["index", "smiles"]
        assert len(frame) == len(DEFAULT_CORPUS)

    def test_novel_recipe_writes_novelty_leaderboard(self, tmp_path, capsys) -> None:
        # 100x-22-a: the 'novel' recipe mirrors 'druglike' (QED x MW x TPSA)
        # and adds the corpus-novelty objective; the leaderboard gains the
        # per-objective breakdown columns including 'novelty'.
        from pharmaml.cli import main

        out = tmp_path / "ga_novel.csv"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "novel",
            "--iterations", "3", "--population", "16", "--seed", "42",
            "--top", "3", "--out", str(out),
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "method   : ga" in output
        assert "novelty" in output
        assert out.is_file()
        frame = pd.read_csv(out)
        assert list(frame.columns) == [
            "rank", "smiles", "score", "qed", "mw_band", "tpsa_band", "novelty",
        ]
        assert len(frame) > 0
        assert frame["novelty"].between(0.0, 1.0).all()

    def test_scaffold_flag_runs_seeded_ga(self, capsys) -> None:
        # 100x-22-a: --scaffold switches the GA to decoration resampling;
        # the run reports the mode.
        from pharmaml.cli import main

        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed", "--scaffold",
            "--seed-smiles", ASPIRIN,
            "--iterations", "3", "--population", "16", "--seed", "42", "--top", "3",
        ])
        assert code == 0
        output = capsys.readouterr().out
        assert "scaffold" in output
        assert "genetic_algorithm" in output

    def test_scaffold_flag_requires_ga_method(self, capsys) -> None:
        # --scaffold is a GA mutation mode: asking an RL run for it must be
        # an actionable typed error, not a silently ignored flag.
        from pharmaml.cli import main

        code = main([
            "generative", "run",
            "--method", "rl", "--recipe", "qed", "--scaffold",
            "--iterations", "1",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "--scaffold" in error and "ga" in error

    def test_run_help_documents_scaffold_and_novel(self, capsys) -> None:
        from pharmaml.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main(["generative", "run", "--help"])
        assert excinfo.value.code == 0
        output = capsys.readouterr().out
        assert "--scaffold" in output
        assert "novel" in output

    def test_run_help_documents_every_flag(self, capsys) -> None:
        from pharmaml.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main(["generative", "run", "--help"])
        assert excinfo.value.code == 0
        output = capsys.readouterr().out
        for flag in (
            "--method", "--recipe", "--seeds", "--seed-smiles", "--iterations",
            "--population", "--batch-size", "--prior-epochs", "--seed",
            "--top", "--out", "--plot",
        ):
            assert flag in output, flag


# ---------------------------------------------------------------------------
# The novelty objective (100x-22-a)
# ---------------------------------------------------------------------------
class TestNoveltyObjective:
    def test_corpus_member_scores_zero(self) -> None:
        # A molecule present in the reference corpus is by definition not
        # novel: max Tanimoto = 1 -> score exactly 0.0.
        objective = NoveltyObjective()
        for member in (ASPIRIN, PARACETAMOL, CAFFEINE):
            assert member in DEFAULT_CORPUS
            assert objective.score(member) == pytest.approx(0.0, abs=1e-12), member

    def test_structurally_distant_molecule_scores_high(self) -> None:
        # Probed against rdkit 2026.09.1 with the default corpus / radius 2 /
        # 2048 bits: cholesterol max-Tanimoto 0.358 -> 0.642, squalene 0.149
        # -> 0.851.  Both pinned with margin around the probe.
        objective = NoveltyObjective()
        assert objective.score(CHOLESTEROL) == pytest.approx(0.6418, abs=0.05)
        assert objective.score(SQUALENE) == pytest.approx(0.8511, abs=0.05)
        # the spec contract: a distant molecule scores well above 0.3
        assert objective.score(CHOLESTEROL) > 0.3
        for smiles in (CHOLESTEROL, SQUALENE):
            assert 0.0 <= objective.score(smiles) <= 1.0

    def test_custom_corpus_changes_the_reference(self) -> None:
        # Aspirin-only corpus: paracetamol's Tanimoto to aspirin is the
        # pinned 0.2222 (see test_similarity_aspirin_paracetamol_is_pinned)
        # -> novelty 0.7778; aspirin itself is a member -> 0.0.
        objective = NoveltyObjective(corpus=[ASPIRIN])
        assert objective.score(PARACETAMOL) == pytest.approx(0.777778, abs=1e-3)
        assert objective.score(ASPIRIN) == pytest.approx(0.0, abs=1e-12)
        # widening the corpus can only *lower* novelty (more references)
        assert (
            NoveltyObjective(corpus=[ASPIRIN]).score(PARACETAMOL)
            >= NoveltyObjective(corpus=[ASPIRIN, PARACETAMOL]).score(PARACETAMOL)
        )

    def test_default_corpus_is_used_and_sanitized(self) -> None:
        # corpus=None -> DEFAULT_CORPUS; entries are canonicalised and
        # de-duplicated (duplicate spellings must not skew anything).
        objective = NoveltyObjective()
        assert objective.corpus == tuple(DEFAULT_CORPUS)
        assert len(objective.corpus) == len(DEFAULT_CORPUS) == 244
        duplicated = NoveltyObjective(corpus=[ASPIRIN, ASPIRIN, f" {PARACETAMOL} "])
        assert duplicated.corpus == (ASPIRIN, PARACETAMOL)

    def test_empty_or_all_invalid_corpus_raises(self) -> None:
        with pytest.raises(DataValidationError, match="corpus is empty"):
            NoveltyObjective(corpus=[])
        with pytest.raises(DataValidationError, match="no valid SMILES"):
            NoveltyObjective(corpus=[INVALID, ""])

    def test_duck_typed_corpus_misuse_is_a_typed_error(self) -> None:
        # A bare string would silently iterate CHARACTERS ("CCO" -> corpus of
        # C, C, O atoms); a non-sequence crashes with a raw TypeError.  Both
        # must surface as the typed DataValidationError instead.
        with pytest.raises(DataValidationError, match="sequence of SMILES"):
            NoveltyObjective(corpus="CCO")  # type: ignore[arg-type]
        with pytest.raises(DataValidationError, match="sequence of SMILES"):
            NoveltyObjective(corpus=42)  # type: ignore[arg-type]

    def test_non_string_corpus_entries_warn_when_dropped(self, caplog) -> None:
        # Docstring contract (100x-22-b fail-first): "invalid entries are
        # dropped with a warning".  Unparseable STRINGS warned; non-string
        # entries were dropped SILENTLY -- silent data loss.  They must warn
        # too, and the valid entries must still survive.
        import logging as _logging

        # TWO independent silencers must be lifted for caplog to see the
        # record (established suite convention, cf. test_descriptors.py's
        # _visible_logs fixture): sibling modules call
        # logging.disable(CRITICAL) at import time, and CLI tests call
        # configure_logging() which sets propagate=False on the pharmaml
        # root logger while pytest's caplog handler lives on the true root.
        prev_disable = _logging.root.manager.disable
        pharma_root = _logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        _logging.disable(_logging.NOTSET)
        pharma_root.propagate = True
        try:
            with caplog.at_level(_logging.WARNING, logger="pharmaml.generative.scoring"):
                objective = NoveltyObjective(corpus=[42, None, ASPIRIN])
        finally:
            pharma_root.propagate = prev_propagate
            _logging.disable(prev_disable)
        assert objective.corpus == (ASPIRIN,)  # the valid entry survives
        warnings_seen = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == _logging.WARNING
        ]
        assert any("non-string" in message for message in warnings_seen), warnings_seen

    def test_invalid_smiles_scores_zero_with_accounting(self) -> None:
        # Mirrors the sibling-objective contract (see QedObjective): invalid
        # input scores 0.0 and is counted, never raised.
        objective = NoveltyObjective(corpus=[ASPIRIN])
        assert objective.score(INVALID) == 0.0
        assert objective.score("") == 0.0
        assert objective.n_invalid == 2
        assert objective.n_scored == 2
        objective.reset_stats()
        assert objective.n_invalid == 0 and objective.n_scored == 0

    def test_works_inside_multi_objective_with_veto(self) -> None:
        objective = MultiObjective(
            [NoveltyObjective(corpus=[ASPIRIN]), QedObjective()],
            weights=[1.0, 1.0],
        )
        total, breakdown = objective.score_with_breakdown(PARACETAMOL)
        assert set(breakdown) == {"novelty", "qed"}
        assert breakdown["novelty"] == pytest.approx(0.777778, abs=1e-3)
        assert breakdown["qed"] == pytest.approx(0.595026, abs=1e-4)
        assert total == pytest.approx((0.777778 * 0.595026) ** 0.5, rel=1e-3)
        # a corpus member vetoes the whole molecule (novelty 0 -> total 0)
        assert objective.score(ASPIRIN) == 0.0
        assert objective.score(INVALID) == 0.0

    def test_same_input_gives_the_same_score(self) -> None:
        batch = [ASPIRIN, PARACETAMOL, CHOLESTEROL, SQUALENE, "CCO", INVALID]
        first = NoveltyObjective().score_many(batch)
        second = NoveltyObjective().score_many(batch)
        assert first == second

    def test_constructor_validation(self) -> None:
        for kwargs in ({"radius": -1}, {"radius": 1.5}, {"radius": "2"},
                       {"n_bits": 16}, {"n_bits": 2048.5}):
            with pytest.raises(DataValidationError):
                NoveltyObjective(corpus=[ASPIRIN], **kwargs)  # type: ignore[arg-type]
        # sane alternative fingerprints are accepted and stay in range
        objective = NoveltyObjective(corpus=[ASPIRIN], radius=1, n_bits=1024)
        assert 0.0 <= objective.score(PARACETAMOL) <= 1.0


# ---------------------------------------------------------------------------
# Scaffold-preserving mutation (100x-22-a)
# ---------------------------------------------------------------------------
def _murcko(smiles: str) -> str:
    """Canonical Murcko framework SMILES of ``smiles`` (test helper)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def _reassemble(core_smi: str, frag_smis: list[str]) -> str:
    """molzip(core, decorations) -> canonical SMILES (round-trip helper)."""
    from rdkit import Chem

    core = Chem.MolFromSmiles(core_smi)
    side = Chem.MolFromSmiles(".".join(frag_smis))
    zipped = Chem.molzip(core, side)
    Chem.SanitizeMol(zipped)
    return Chem.MolToSmiles(zipped)


class TestScaffoldMutation:
    def test_decoration_pool_is_built_from_the_corpus(self) -> None:
        # Probed against rdkit 2026.09.1: the 244-entry corpus decomposes into
        # 65 unique single-dummy decorations (acyclic only -- a ring in a
        # decoration would be part of the Murcko framework by definition).
        pool = _build_decoration_pool(list(DEFAULT_CORPUS))
        assert len(pool) >= 60
        assert len(set(pool)) == len(pool)  # de-duplicated
        assert _build_decoration_pool(list(DEFAULT_CORPUS)) == pool  # deterministic

        from rdkit import Chem

        for entry in pool:
            assert entry.count("*") == 1, entry  # exactly one dummy atom
            frag = Chem.MolFromSmiles(entry)
            assert frag is not None, entry
            dummies = [a for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
            assert len(dummies) == 1, entry
            assert frag.GetRingInfo().NumRings() == 0, entry  # acyclic vocabulary

    def test_decoration_pool_skips_acyclic_and_bare_scaffolds(self) -> None:
        # No Murcko scaffold (acyclic) or no decorations (bare ring systems)
        # contribute nothing to the pool.
        assert _build_decoration_pool(["CCO", "CCN", "c1ccccc1"]) == []

    def test_scaffold_split_round_trips_the_parent(self) -> None:
        # core (attachment-labelled dummies) + decorations re-zip into exactly
        # the parent molecule: the decomposition loses nothing.
        for parent in (ASPIRIN, PARACETAMOL, CAFFEINE, NICOTINE,
                       "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"):  # warfarin
            split = _scaffold_split(parent)
            assert split is not None, parent
            core_smi, frag_smis = split
            assert "[*:" in core_smi
            assert len(frag_smis) >= 1
            assert _reassemble(core_smi, list(frag_smis)) == parent, parent

    def test_scaffold_mutate_preserves_the_parent_scaffold(self) -> None:
        # The operator contract: every SUCCESSFUL decoration-resampling mutant
        # keeps the parent's Murcko framework verbatim (structural guarantee:
        # corpus decorations are acyclic, so they can never add a ring).
        import random

        pool = _build_decoration_pool(list(DEFAULT_CORPUS))
        rng = random.Random(123)
        for parent in (ASPIRIN, PARACETAMOL, CAFFEINE, NICOTINE):
            parent_scaffold = _murcko(parent)
            mutants = [
                _scaffold_mutate(parent, pool, rng, max_smiles_length=120)
                for _ in range(30)
            ]
            successes = [m for m in mutants if m is not None]
            # the operator almost always fires for drug-like ring parents;
            # occasional RDKit failures fall back (returned as None)
            assert len(successes) >= 24, (parent, len(successes))
            for mutant in successes:
                assert _murcko(mutant) == parent_scaffold, (parent, mutant)
            # and it actually explores (not just cloning the parent back)
            assert len(set(successes)) >= 2, parent

    def test_scaffold_mutate_returns_none_for_acyclic_or_bare_scaffold(self) -> None:
        # Acyclic parents have no Murcko framework and bare ring systems have
        # no decorations: the caller must fall back to character mutation.
        import random

        pool = _build_decoration_pool([ASPIRIN])
        rng = random.Random(0)
        for parent in ("CCO", "c1ccccc1", INVALID):
            assert _scaffold_mutate(parent, pool, rng, max_smiles_length=120) is None, parent

    def test_scaffold_mutate_is_deterministic(self) -> None:
        import random

        pool = _build_decoration_pool(list(DEFAULT_CORPUS))
        first = [
            _scaffold_mutate(ASPIRIN, pool, random.Random(7), max_smiles_length=120)
            for _ in range(40)
        ]
        second = [
            _scaffold_mutate(ASPIRIN, pool, random.Random(7), max_smiles_length=120)
            for _ in range(40)
        ]
        assert first == second

    def test_seeded_scaffold_run_preserves_the_seed_scaffold(self) -> None:
        # Lead-optimisation property: with the seed molecule as the only
        # corpus entry, >= 95% of every unique molecule the GA produces keeps
        # the aspirin seed's Murcko framework (benzene) verbatim.
        result = GeneticOptimizer(
            QedObjective(),
            population_size=16,
            generations=3,
            seed=42,
            mutation="scaffold",
        ).optimize(seed_smiles=[ASPIRIN], corpus=[ASPIRIN])
        assert len(result) >= 2
        benzene = _murcko(ASPIRIN)
        preserved = sum(1 for smi in result.smiles if _murcko(smi) == benzene)
        assert preserved / len(result.smiles) >= 0.95
        # deterministic: a second identical run reproduces the leaderboard
        rerun = GeneticOptimizer(
            QedObjective(),
            population_size=16,
            generations=3,
            seed=42,
            mutation="scaffold",
        ).optimize(seed_smiles=[ASPIRIN], corpus=[ASPIRIN])
        assert rerun.smiles == result.smiles
        assert rerun.scores == result.scores

    def test_scaffold_mode_smoke_on_the_default_corpus(self) -> None:
        result = GeneticOptimizer(
            QedObjective(),
            population_size=16,
            generations=3,
            seed=42,
            mutation="scaffold",
        ).optimize()
        assert len(result) > 0
        assert result.metadata["mutation"] == "scaffold"
        for smiles in result.smiles:
            assert canonical_smiles(smiles) == smiles  # valid + canonical

    def test_acyclic_seed_falls_back_without_crashing(self) -> None:
        # A corpus with no ring systems yields an empty decoration pool, so
        # scaffold mode degrades to the standard character mutation -- the
        # run must still complete and produce valid molecules.
        result = GeneticOptimizer(
            QedObjective(),
            population_size=12,
            generations=2,
            seed=42,
            mutation="scaffold",
        ).optimize(seed_smiles=["CCO"], corpus=["CCO", "CCN"])
        assert len(result) > 0
        assert all(canonical_smiles(smi) is not None for smi in result.smiles)

    def test_mutation_parameter_validation_and_default(self) -> None:
        with pytest.raises(DataValidationError, match="mutation"):
            GeneticOptimizer(QedObjective(), mutation="bogus")
        # the default is unchanged (back-compat): uniform character mutation
        optimizer = GeneticOptimizer(QedObjective())
        assert optimizer.mutation == "uniform"
        result = optimizer.optimize(corpus=["CCO", "CCN"], seed_smiles=["CCO"])
        assert result.metadata["mutation"] == "uniform"

    def test_default_mutation_mode_results_are_unchanged(self) -> None:
        # Back-compat pins recorded BEFORE the 100x-22-a changes (see
        # /tmp/task22a/backcompat.txt): the uniform-mode GA must reproduce
        # them molecule-for-molecule, generation score for generation score.
        result = GeneticOptimizer(
            QedObjective(), population_size=32, generations=5, seed=42
        ).optimize()
        assert result.best == "COc1ccc2[nH]c(CCC(C)=O)cc2c1"
        assert result.best_score == pytest.approx(0.855097, abs=1e-6)
        # (r263 environment: rdkit 2026.03.4 canonicalises two near-dupe
        #  children to equal strings, so the unique set is 135, not the
        #  137 of the pre-upgrade environment; best molecule, best score
        #  and the full generation history are unchanged)
        # (r265 environment: rdkit 2026.09.1 keeps the two near-dupes
        #  distinct again — 137, matching the pre-r263 count; best
        #  molecule and best score unchanged)
        assert len(result) == 137

        multi = MultiObjective(
            [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
        )
        multi_result = GeneticOptimizer(
            multi, population_size=24, generations=4, seed=42
        ).optimize()
        assert multi_result.best == "CC(=O)Nc1ccbcc2[nH]cnc2ccc1"
        assert multi_result.best_score == pytest.approx(0.839569, abs=1e-6)
        assert [round(entry["best_score"], 6) for entry in multi_result.history] == [
            0.833081, 0.833081, 0.839569, 0.839569,
        ]


class TestScaffoldHopCrossover:
    """BREED-style homologous-cut crossover (round 100x-27 flagship).

    Covers the standalone :func:`~pharmaml.generative.scaffold_hop_crossover`
    operator (validity, determinism, composition bounds, framework hops,
    fallback contract, length cap) and its wiring into
    :class:`~pharmaml.generative.GeneticOptimizer` (mode validation,
    determinism, metadata reporting, the full structural mode).
    """

    PARENT_A = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    PARENT_B = "NC(=O)c1ccccc1"  # benzamide

    @staticmethod
    def _murcko(smiles: str) -> str:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = rdkit.Chem.MolFromSmiles(smiles)
        assert mol is not None
        return rdkit.Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

    def test_children_valid_deterministic_and_bounded(self) -> None:
        for seed in range(12):
            child = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_B, random.Random(seed)
            )
            assert child is not None
            again = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_B, random.Random(seed)
            )
            assert again == child  # pure function of (parents, rng state, cap)
            mol = rdkit.Chem.MolFromSmiles(child)
            assert mol is not None
            # no attachment dummy ever leaks into an offspring
            assert all(atom.GetAtomicNum() != 0 for atom in mol.GetAtoms())
            # composition: one fragment from each parent, so strictly
            # fewer heavy atoms than both parents combined
            n_a = rdkit.Chem.MolFromSmiles(self.PARENT_A).GetNumAtoms()
            n_b = rdkit.Chem.MolFromSmiles(self.PARENT_B).GetNumAtoms()
            assert mol.GetNumAtoms() < n_a + n_b
            # element provenance: only elements present in the parents
            elems = (
                {a.GetSymbol() for a in rdkit.Chem.MolFromSmiles(self.PARENT_A).GetAtoms()}
                | {a.GetSymbol() for a in rdkit.Chem.MolFromSmiles(self.PARENT_B).GetAtoms()}
            )
            assert {a.GetSymbol() for a in mol.GetAtoms()} <= elems

    def test_framework_hops_both_parents_scaffolds(self) -> None:
        # 2-methylnaphthalene x methyl-pyridine-acid: the naphthalene and
        # pyridine fragments recombine into frameworks neither parent has
        parent_a = "Cc1ccc2ccccc2c1"
        parent_b = "Cc1ccncc1C(=O)O"
        scaff_a = self._murcko(parent_a)
        scaff_b = self._murcko(parent_b)
        assert scaff_a != scaff_b
        hopped = 0
        for seed in range(24):
            child = scaffold_hop_crossover(parent_a, parent_b, random.Random(seed))
            assert child is not None
            if self._murcko(child) not in (scaff_a, scaff_b):
                hopped += 1
        assert hopped >= 12  # probe on rdkit 2026.09.1: 24/24 seeds hop

    def test_no_homologous_bridge_returns_none(self) -> None:
        # benzene x pyridine: no acyclic single bonds to cut at all
        assert scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None
        # biphenyl x diphenyl ether: bridges exist on both sides but the
        # join signatures cannot match (a C-C bridge never crosses C-O)
        assert (
            scaffold_hop_crossover(
                "c1ccccc1-c1ccccc1", "c1ccccc1Oc1ccccc1", random.Random(0)
            )
            is None
        )

    def test_invalid_or_disconnected_parents_return_none(self) -> None:
        assert scaffold_hop_crossover("not_a_smiles", "CCO", random.Random(0)) is None
        assert scaffold_hop_crossover("CCO", "not_a_smiles", random.Random(0)) is None
        # a dot-disconnected parent has no recombination semantics
        assert scaffold_hop_crossover("CCO.CC", "CCO", random.Random(0)) is None
        assert scaffold_hop_crossover("CCO", "CCO.CC", random.Random(0)) is None

    def test_length_cap_enforced_with_sibling_fallback(self) -> None:
        # seed 0's drawn pairing is the 26-char hybrid: over the cap, so
        # the 12-char sibling pairing is returned instead
        small = scaffold_hop_crossover(
            self.PARENT_A, self.PARENT_B, random.Random(0), max_smiles_length=20
        )
        assert small == "NC(=O)C(=O)O"
        # biindole parents: EVERY pairing is the same 30+-char biindole,
        # so the floor cap leaves no viable child
        biindole = "c1ccc2[nH]ccc2c1-c1ccc2[nH]ccc2c1"
        for seed in range(4):
            assert (
                scaffold_hop_crossover(
                    biindole, biindole, random.Random(seed), max_smiles_length=20
                )
                is None
            )

    def test_length_cap_floor_raises(self) -> None:
        with pytest.raises(ValueError, match="max_smiles_length"):
            scaffold_hop_crossover(
                "CCO", "CCO", random.Random(0), max_smiles_length=10
            )

    def test_self_crossover_yields_valid_molecules(self) -> None:
        # cutting a molecule with itself at different loci legitimately
        # builds duplication hybrids (larger than the parent) -- they must
        # still sanitise and respect the cap
        for seed in range(6):
            child = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_A, random.Random(seed)
            )
            assert child is None or rdkit.Chem.MolFromSmiles(child) is not None

    # -- GeneticOptimizer wiring -----------------------------------------
    def test_crossover_mode_validation_and_default(self) -> None:
        with pytest.raises(DataValidationError, match="crossover"):
            GeneticOptimizer(QedObjective(), crossover="breed")
        optimizer = GeneticOptimizer(QedObjective())
        assert optimizer.crossover == "splice"  # back-compat default

    def test_hop_mode_runs_deterministic_and_reports_mode(self) -> None:
        objective = MultiObjective(
            [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3]
        )

        def run() -> GenerationResult:
            return GeneticOptimizer(
                objective, population_size=24, generations=4, seed=42,
                crossover="scaffold_hop",
            ).optimize()

        first, second = run(), run()
        assert len(first) > 0
        assert first.smiles == second.smiles  # deterministic given the seed
        assert first.metadata["crossover"] == "scaffold_hop"
        assert first.metadata["mutation"] == "uniform"
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)

    def test_structural_mode_scaffold_mutation_with_hop_crossover(self) -> None:
        # the full structural mode: framework recombination + decoration
        # resampling, seeded from aspirin over a tiny corpus
        result = GeneticOptimizer(
            QedObjective(), population_size=24, generations=4, seed=7,
            mutation="scaffold", crossover="scaffold_hop",
        ).optimize(
            seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"],
            corpus=["CC(=O)Oc1ccccc1C(=O)O", "NC(=O)c1ccccc1", "c1ccc(O)cc1"],
        )
        assert len(result) > 0
        assert result.metadata["mutation"] == "scaffold"
        assert result.metadata["crossover"] == "scaffold_hop"
        assert all(canonical_smiles(smi) is not None for smi in result.smiles)

    def test_run_genetic_optimization_passes_crossover_through(self) -> None:
        result = run_genetic_optimization(
            QedObjective(), generations=2, seed=42, crossover="scaffold_hop"
        )
        assert result.metadata["crossover"] == "scaffold_hop"


class TestScaffoldHopRingCuts:
    """Ring-bond cuts (round 100x-28): merge + transplant geometries.

    The BREED crossover historically cut *acyclic bridge* bonds only, so
    ring *topology* never recombined.  The two new geometries:

    * **merge** -- one single ring bond cut per parent, the two opened
      rings zipped into one combined ring holding atoms from both;
    * **transplant** -- one single ring bond cut in the ring parent, a
      two-ended linker atom (acyclic, two single bonds, homologous
      flanks) excised from the other parent and zipped into the cut:
      ring expansion / homologation with one atom from the other parent.
    """

    @staticmethod
    def _ring_sizes(smiles: str) -> list[int]:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        assert mol is not None
        return sorted(len(ring) for ring in mol.GetRingInfo().AtomRings())

    def test_ring_merge_combines_both_parents_rings(self) -> None:
        # cyclohexane x piperidine: every atom is in a ring, so the only
        # viable geometry is the merge -> a single 12-ring, C + N, from
        # both parents, deterministic per rng state
        for seed in range(6):
            child = scaffold_hop_crossover(
                "C1CCCCC1", "C1CCNCC1", random.Random(seed)
            )
            assert child is not None
            assert self._ring_sizes(child) == [12]
            mol = rdkit.Chem.MolFromSmiles(child)
            assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C", "N"}
            again = scaffold_hop_crossover(
                "C1CCCCC1", "C1CCNCC1", random.Random(seed)
            )
            assert again == child

    def test_ring_transplant_inserts_linker_atom(self) -> None:
        # piperidine x diethyl ether: the ether oxygen is transplanted
        # into a cut piperidine C-C bond -> a 7-ring oxazepane keeping
        # the N (from A) and gaining the O (from B)
        for seed in range(12):
            child = scaffold_hop_crossover(
                "C1CCNCC1", "CCOCC", random.Random(seed)
            )
            assert child is not None
            assert self._ring_sizes(child) == [7]
            mol = rdkit.Chem.MolFromSmiles(child)
            syms = {a.GetSymbol() for a in mol.GetAtoms()}
            assert syms == {"C", "N", "O"}  # provenance: A + B
            assert all(a.GetAtomicNum() != 0 for a in mol.GetAtoms())
            again = scaffold_hop_crossover(
                "C1CCNCC1", "CCOCC", random.Random(seed)
            )
            assert again == child

    def test_transplant_works_in_both_argument_orders(self) -> None:
        # ring parent first or second: the scan covers both orientations
        first = scaffold_hop_crossover("CCOCC", "C1CCNCC1", random.Random(3))
        second = scaffold_hop_crossover("C1CCNCC1", "CCOCC", random.Random(3))
        assert first is not None and second is not None
        assert self._ring_sizes(first) == [7]
        assert self._ring_sizes(second) == [7]

    def test_transplant_from_thioether_and_amine_linkers(self) -> None:
        # S linker -> 7-ring thiazepane; NH linker -> azepane: the
        # geometry is linker-agnostic (any degree-2 acyclic atom with
        # homologous flanks)
        child_s = scaffold_hop_crossover("C1CCNCC1", "CCSCC", random.Random(0))
        child_n = scaffold_hop_crossover("C1CCCCC1", "CCNCC", random.Random(0))
        assert child_s is not None and self._ring_sizes(child_s) == [7]
        assert child_n is not None and self._ring_sizes(child_n) == [7]
        assert "S" in {a.GetSymbol() for a in rdkit.Chem.MolFromSmiles(child_s).GetAtoms()}

    def test_aromatic_ring_bonds_are_never_cut(self) -> None:
        # benzene's ring bonds are aromatic: not single-bond candidates,
        # and benzene has no acyclic degree-2 linker either -> no geometry
        assert scaffold_hop_crossover("c1ccccc1", "C1CCNCC1", random.Random(0)) is None
        # purely aromatic parents stay on the classic bridge geometry
        assert scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None

    def test_ring_merge_respects_length_cap(self) -> None:
        # two cyclododecanes: the only geometry is the 24-ring merge whose
        # canonical SMILES (26 chars) cannot fit the 20-char floor cap
        cyclododecane = "C1CCCCCCCCCCCC1"
        for seed in range(4):
            assert (
                scaffold_hop_crossover(
                    cyclododecane,
                    cyclododecane,
                    random.Random(seed),
                    max_smiles_length=20,
                )
                is None
            )

    def test_ga_crossover_reaches_ring_cut_children(self) -> None:
        # unit level: the GA's _crossover wires the operator's new ring
        # geometries through (fallback contract: parent_a when None)
        optimizer = GeneticOptimizer(QedObjective(), crossover="scaffold_hop")
        child = optimizer._crossover("C1CCNCC1", "CCOCC", random.Random(5))
        assert child is not None
        assert self._ring_sizes(child) == [7]

    def test_ga_hop_mode_absorbs_ring_parents_deterministically(self) -> None:
        # corpus with a saturated ring + an acyclic linker donor: the
        # structural GA runs (new geometries reachable) and stays a pure
        # function of the seed
        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(), population_size=16, generations=3, seed=11,
                crossover="scaffold_hop",
            ).optimize(seed_smiles=["C1CCNCC1"], corpus=["C1CCNCC1", "CCOCC"])

        first, second = run(), run()
        assert first.smiles == second.smiles
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)

    def test_dummy_bearing_parents_are_rejected_not_transplanted(self) -> None:
        # a parent containing a bare dummy atom (*, atomic number 0) is
        # parseable and connected -- but a dummy is NOT a transplantable
        # linker: no child may ever carry an attachment point (round
        # 100x-28 review finding 1)
        for parents in (
            ("C1CCNCC1", "CC*CC"),
            ("CC*CC", "C1CCNCC1"),
            ("C1CCCCC1", "*CC*"),
        ):
            for seed in range(4):
                child = scaffold_hop_crossover(*parents, random.Random(seed))
                assert child is None, (parents, seed, child)

    def test_flipped_merge_candidate_assembles_homologously(self) -> None:
        # piperidine x morpholine, seed 0: the drawn candidate is a
        # flipped-only match (cut bonds stored N-C vs C-N) -- the merge
        # must then join N-N and C-C seams (the homologous orientation),
        # visible as an N-N bond in the merged ring; joining N-C / C-N
        # would violate the module's signature-matching invariant
        # (round 100x-28 review finding 2).  Probe on rdkit 2026.09.1.
        child = scaffold_hop_crossover("C1CCNCC1", "C1COCCN1", random.Random(0))
        assert child is not None
        assert self._ring_sizes(child) == [12]
        mol = rdkit.Chem.MolFromSmiles(child)
        has_nn_bond = any(
            {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {7}
            for b in mol.GetBonds()
        )
        assert has_nn_bond, f"non-homologous seam assembly: {child}"
        again = scaffold_hop_crossover("C1CCNCC1", "C1COCCN1", random.Random(0))
        assert again == child  # the flip is deterministic, not accidental


class TestScaffoldHopPathTransplants:
    """Multi-atom linker transplants (round 100x-29): excise a *path*.

    The 100x-28 transplant geometry excised a single two-ended linker
    atom (an ether O, a methylene).  A whole short acyclic *path* -- an
    ethylene bridge ``-CH2CH2-``, an oxyethylene ``-OCH2-``, a diethylene
    ``-OCH2CH2O-`` -- is the same move with more atoms: the path is cut
    loose from its two hosts and zipped into a cut ring bond of the other
    parent, homologously (each seam atom must match its counterpart's
    signature).  Ring expansion by 2-4 atoms: piperidine -> azocane,
    piperidine + diethylene glycol -> 8/9-rings -- the ring-size axis of
    scaffold morphing.  Flipped-only matches (path hosts matching the cut
    bond's endpoints in reverse order) must excise the *reversed* path,
    exactly like the flipped merge swaps dummy labels (100x-28 finding 2).
    """

    @staticmethod
    def _ring_sizes(smiles: str) -> list[int]:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        assert mol is not None
        return sorted(len(ring) for ring in mol.GetRingInfo().AtomRings())

    def test_two_atom_path_expands_ring_to_eight(self) -> None:
        # piperidine x hexane: hexane's middle -CH2CH2- has homologous
        # degree-2 CH2 hosts, so it transplants into a cut piperidine
        # C-C bond -> an 8-ring azocane; the single-atom CH2 transplants
        # (7-ring azepanes) stay available too
        observed = set()
        for seed in range(18):
            child = scaffold_hop_crossover(
                "C1CCNCC1", "CCCCCC", random.Random(seed)
            )
            assert child is not None, seed
            sizes = self._ring_sizes(child)
            assert sizes in ([7], [8]), (seed, child, sizes)
            observed.update(sizes)
            mol = rdkit.Chem.MolFromSmiles(child)
            assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C", "N"}
        assert observed == {7, 8}  # the new geometry is actually reachable
        again = scaffold_hop_crossover(
            "C1CCNCC1", "CCCCCC", random.Random(5)
        )
        assert again is not None and self._ring_sizes(again) == [8]

    def test_hetero_path_transplants_insert_oxygen_chains(self) -> None:
        # piperidine x diethylene glycol: -CH2O- paths -> 8-rings,
        # -CH2OCH2- paths -> 9-rings, the single O -> 7-ring: the ring
        # grows along the linker alphabet, O always inside the ring
        observed = set()
        for seed in range(30):
            child = scaffold_hop_crossover(
                "C1CCNCC1", "OCCOCCO", random.Random(seed)
            )
            assert child is not None, seed
            sizes = self._ring_sizes(child)
            assert sizes in ([7], [8], [9]), (seed, child, sizes)
            observed.update(sizes)
            mol = rdkit.Chem.MolFromSmiles(child)
            assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C", "N", "O"}
        assert observed == {7, 8, 9}
        again = scaffold_hop_crossover(
            "C1CCNCC1", "OCCOCCO", random.Random(0)
        )
        assert again is not None and self._ring_sizes(again) == [8]

    def test_path_length_is_capped_at_four_atoms(self) -> None:
        # triethylene glycol has a 6-atom maximal chain: the 5-atom
        # middle path WOULD be homologous, but excisions are capped at
        # four atoms, so no child may carry an 11-ring (6 + 5)
        for seed in range(40):
            child = scaffold_hop_crossover(
                "C1CCNCC1", "OCCOCCOCCO", random.Random(seed)
            )
            if child is None:
                continue
            assert max(self._ring_sizes(child)) <= 10, (seed, child)

    def test_linker_path_units_enumeration_is_pinned(self) -> None:
        # white-box: maximal chains -> contiguous sublists of length 2..4,
        # chains in order of their smallest atom, sublists left-to-right;
        # hexane's chain is C2-C3-C4-C5 (methyls are degree-1 hosts)
        from pharmaml.generative.scaffold_hop import _linker_path_units

        units = _linker_path_units(rdkit.Chem.MolFromSmiles("CCCCCC"))
        assert units == [
            ((1, 2), ((6, 1, False, 0), (6, 2, False, 0))),
            ((1, 2, 3), ((6, 1, False, 0), (6, 2, False, 0))),
            ((1, 2, 3, 4), ((6, 1, False, 0), (6, 1, False, 0))),
            ((2, 3), ((6, 2, False, 0), (6, 2, False, 0))),
            ((2, 3, 4), ((6, 2, False, 0), (6, 1, False, 0))),
            ((3, 4), ((6, 2, False, 0), (6, 1, False, 0))),
        ]
        # diethylene glycol: one 5-atom chain O-adjacent hosts included
        deg = _linker_path_units(rdkit.Chem.MolFromSmiles("OCCOCCO"))
        assert len(deg) == 9
        assert all(len(path) <= 4 for path, _ in deg)
        # a unit crossing a double bond (propanal's C-C(=O)) never excises
        propanal = _linker_path_units(rdkit.Chem.MolFromSmiles("CCC=O"))
        assert propanal == []

    def test_excise_linker_path_builds_zippable_donor(self) -> None:
        # white-box: the donor is [*:1]-path-[*:2] with map-numbered
        # dummies at both ends; n == 1 degenerates to the atom linker
        from pharmaml.generative.scaffold_hop import _excise_linker_path

        deg = rdkit.Chem.MolFromSmiles("OCCOCCO")
        excision = _excise_linker_path(deg, (2, 3))  # -CH2-O-
        assert excision is not None
        donor, host_sigs = excision
        dummies = [a for a in donor.GetAtoms() if a.GetAtomicNum() == 0]
        assert sorted(d.GetAtomMapNum() for d in dummies) == [1, 2]
        # canonical rendering of a mapped-dummy chain (the map numbers
        # break the terminal symmetry, so RDKit branches from the middle)
        assert rdkit.Chem.MolToSmiles(donor) == "C(O[*:2])[*:1]"
        assert host_sigs == ((6, 2, False, 0), (6, 2, False, 0))
        # single-atom path: hosts in neighbour order, like _excise_linker_atom
        single = _excise_linker_path(deg, (3,))
        assert single is not None
        assert single[1] == ((6, 2, False, 0), (6, 2, False, 0))
        assert rdkit.Chem.MolToSmiles(single[0]) == "O([*:1])[*:2]"
        # non-adjacent / repeating indices are rejected, never assembled
        assert _excise_linker_path(deg, (1, 3)) is None
        assert _excise_linker_path(deg, (3, 3)) is None

    def test_flipped_path_candidate_assembles_homologously(self) -> None:
        # piperazine x CNCOCCC, seed 3: the drawn candidate is a
        # flipped-only match (path hosts N-then-C vs the cut bond's
        # C-then-N), so the path must be excised REVERSED -- the ether O's
        # donor host was a carbon, so in the child O must bond carbons
        # only (an as-is excision would seam O to N: C1CNCCNOC1)
        child = scaffold_hop_crossover("N1CCNCC1", "CNCOCCC", random.Random(3))
        assert child is not None
        assert child == "C1CNCOCCN1"  # probe on rdkit 2026.09.1
        assert self._ring_sizes(child) == [8]
        mol = rdkit.Chem.MolFromSmiles(child)
        assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C", "N", "O"}
        for bond in mol.GetBonds():
            pair = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
            assert pair != {"O", "N"}, f"non-homologous seam: {child}"
        again = scaffold_hop_crossover("N1CCNCC1", "CNCOCCC", random.Random(3))
        assert again == child  # the flip is deterministic, not accidental

    def test_path_transplant_works_in_both_argument_orders(self) -> None:
        # ring parent first or second: the path scan covers both sides
        first = scaffold_hop_crossover("CNCOCCC", "N1CCNCC1", random.Random(3))
        second = scaffold_hop_crossover("N1CCNCC1", "CNCOCCC", random.Random(3))
        assert first is not None and second is not None
        assert self._ring_sizes(first) == [8]
        assert self._ring_sizes(second) == [8]

    def test_ga_crossover_reaches_path_children(self) -> None:
        # the GA's _crossover wires the path geometry through: seed 5
        # draws a -CH2CH2- transplant (piperidine x hexane -> azocane)
        optimizer = GeneticOptimizer(QedObjective(), crossover="scaffold_hop")
        child = optimizer._crossover("C1CCNCC1", "CCCCCC", random.Random(5))
        assert child is not None
        assert self._ring_sizes(child) == [8]

    def test_ga_hop_mode_absorbs_path_donors_deterministically(self) -> None:
        # corpus with a saturated ring + a multi-atom linker donor: the
        # structural GA stays a pure function of the seed
        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(), population_size=16, generations=3, seed=11,
                crossover="scaffold_hop",
            ).optimize(seed_smiles=["C1CCNCC1"], corpus=["C1CCNCC1", "OCCOCCO"])

        first, second = run(), run()
        assert first.smiles == second.smiles
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)


class TestScaffoldHopFusedRings:
    """Whole-ring excision + splice (round 100x-30): the fuse geometry.

    The 100x-28/29 geometries recombined ring *content* (merge) and
    ring *size* (transplant).  The full Nicolaou move set's missing
    member recombines ring *systems*: a whole ring is excised from one
    parent -- any ring (aromatic ones included) attached to the rest of
    its molecule through exactly two single anchor bonds at two distinct
    ring atoms -- and spliced across an opened ring bond of the other
    parent.  The child is a fused / bridged bicyclic holding one ring
    from each parent: decalin x piperidine -> azadecalin cores,
    indane x piperidine -> benzazocine cores (the aromatic donor ring
    survives intact -- only its two single aryl-CH2 anchor bonds are
    cut).  Rings with zero/one/three+ outside bonds, spiro anchors
    (one atom, two outside bonds) and aromatic or double anchor bonds
    (naphthalene, indole) never excise.  Flipped-only matches (anchor
    hosts matching the cut bond's endpoints in reverse order) swap the
    piece's dummy labels at execution time -- the merge geometry's
    label-swap trick (100x-28 finding 2).
    """

    @staticmethod
    def _ring_sizes(smiles: str) -> list[int]:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        assert mol is not None
        return sorted(len(ring) for ring in mol.GetRingInfo().AtomRings())

    def test_fuse_splices_whole_ring_across_cut_bond(self) -> None:
        # decalin x piperidine, seed 0: the drawn candidate is a fuse --
        # decalin's left 6-ring excised at its two anchor bonds and
        # spliced across a cut piperidine C-C bond -> a bicyclic
        # (6-ring + 8-ring, an azadecalin core), single fragment, no
        # dummy leaked.  Probe on rdkit 2026.09.1.
        child = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "C1CCNCC1", random.Random(0)
        )
        assert child == "C1CCC2CCNCCCC2C1"
        assert self._ring_sizes(child) == [6, 8]
        mol = rdkit.Chem.MolFromSmiles(child)
        assert mol is not None
        assert len(rdkit.Chem.GetMolFrags(mol)) == 1
        assert mol.GetNumAtoms() == 12  # both parents' 6-rings, spliced
        assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C", "N"}
        assert not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        again = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "C1CCNCC1", random.Random(0)
        )
        assert again == child  # pure function of (parents, rng state)

    def test_fuse_aromatic_donor_ring_survives_intact(self) -> None:
        # indane x piperidine, seed 5: the BENZENE ring is the excised
        # donor (its two single aryl-CH2 anchor bonds cut, every
        # aromatic ring bond untouched) -> a benzo-fused azocane core;
        # the child keeps exactly one aromatic 6-ring.  Probe on
        # rdkit 2026.09.1.
        child = scaffold_hop_crossover(
            "C1Cc2ccccc2C1", "C1CCNCC1", random.Random(5)
        )
        assert child == "c1ccc2c(c1)CCCCNC2"
        assert self._ring_sizes(child) == [6, 8]
        mol = rdkit.Chem.MolFromSmiles(child)
        assert mol is not None
        aromatic_rings = [
            ring for ring in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        ]
        assert len(aromatic_rings) == 1  # the donor benzene, intact
        again = scaffold_hop_crossover(
            "C1Cc2ccccc2C1", "C1CCNCC1", random.Random(5)
        )
        assert again == child

    def test_fuse_children_are_always_valid_connected_and_capped(self) -> None:
        # invariant sweep: whatever geometry a seed draws, a non-None
        # child is a parseable, single-fragment, dummy-free molecule
        # under the cap -- the fuse geometry obeys the module contract
        for parent_a, parent_b in (
            ("C1CCC2CCCCC2C1", "C1CCNCC1"),
            ("C1CCC2CCCCC2C1", "C1COCC1"),
            ("C1Cc2ccccc2C1", "C1CCNCC1"),
        ):
            for seed in range(40):
                child = scaffold_hop_crossover(
                    parent_a, parent_b, random.Random(seed)
                )
                if child is None:
                    continue
                assert len(child) <= 120, (parent_a, seed, child)
                mol = rdkit.Chem.MolFromSmiles(child)
                assert mol is not None, (parent_a, seed, child)
                assert len(rdkit.Chem.GetMolFrags(mol)) == 1
                assert not any(
                    a.GetAtomicNum() == 0 for a in mol.GetAtoms()
                ), (parent_a, seed, child)
                assert mol.GetRingInfo().NumRings() >= 1

    def test_ring_units_enumeration_is_pinned(self) -> None:
        # white-box: a ring admits exactly when its total outside-bond
        # count is two, at two distinct anchors, both bonds single.
        # Decalin: BOTH 6-rings (each hangs off the other through two
        # single bonds of the adjacent ring -- the fused-system case);
        # indane: only the benzene (the 5-ring has three outside bonds).
        from pharmaml.generative.scaffold_hop import _ring_units

        decalin = _ring_units(rdkit.Chem.MolFromSmiles("C1CCC2CCCCC2C1"))
        # (r263: rdkit 2026.03.4 enumerates ring atoms in a different
        #  traversal order — same memberships, same anchors, same bond
        #  tuple; pins re-recorded against the live enumeration)
        # (r265: rdkit 2026.09.1 traverses in index order — re-recorded
        #  again; memberships, anchors and bond tuples still identical)
        assert decalin == [
            ((0, 1, 2, 3, 8, 9), 3, 8, ((6, 2, False, 0), (6, 2, False, 0))),
            ((3, 4, 5, 6, 7, 8), 3, 8, ((6, 2, False, 0), (6, 2, False, 0))),
        ]
        indane = _ring_units(rdkit.Chem.MolFromSmiles("C1Cc2ccccc2C1"))
        # (r263: same re-record as the decalin pin above; r265 again)
        assert indane == [
            ((2, 3, 4, 5, 6, 7), 2, 7, ((6, 2, False, 0), (6, 2, False, 0))),
        ]
        # rejections: attachment count, spiro anchor, aromatic fusion
        for smi in (
            "C1CCC2(CC1)CCCC2",  # spiro: one anchor, two outside bonds
            "CC1(C)CCCCC1(C)C",  # tri-substituted: three outside bonds
            "c1ccc2ccccc2c1",  # naphthalene: aromatic anchor bonds
            "c1ccc2[nH]ccc2c1",  # indole: aromatic fusion seam
            "Cc1ccccc1",  # mono-substituted: one outside bond
            "C1CCNCC1",  # plain ring: zero outside bonds
        ):
            assert _ring_units(rdkit.Chem.MolFromSmiles(smi)) == [], smi

    def test_excise_ring_builds_zippable_piece(self) -> None:
        # white-box: the piece is the intact ring with map-numbered
        # dummies at the two anchors; host signatures returned in
        # (a1, a2) order for the homology check
        from pharmaml.generative.scaffold_hop import _excise_ring

        decalin = rdkit.Chem.MolFromSmiles("C1CCC2CCCCC2C1")
        excision = _excise_ring(decalin, (0, 1, 2, 3, 8, 9), 3, 8)
        assert excision is not None
        piece, host_sigs = excision
        dummies = [a for a in piece.GetAtoms() if a.GetAtomicNum() == 0]
        assert sorted(d.GetAtomMapNum() for d in dummies) == [1, 2]
        assert piece.GetNumAtoms() == 8  # six ring atoms + two dummies
        assert rdkit.Chem.MolToSmiles(piece) == "C1CCC([*:2])C([*:1])C1"
        assert host_sigs == ((6, 2, False, 0), (6, 2, False, 0))
        # robustness contract: garbage ring/anchor tuples never assemble
        assert _excise_ring(decalin, (0, 1, 2, 3, 8, 9), 3, 3) is None
        assert _excise_ring(decalin, (0, 1, 2, 3, 8, 9), 4, 8) is None
        assert _excise_ring(decalin, (0, 1, 2, 3, 8, 8), 3, 8) is None
        assert _excise_ring(decalin, (0, 1, 2, 3, 8), 3, 8) is None

    def test_flipped_fuse_candidate_assembles_homologously(self) -> None:
        # asymmetric donor (ring O breaks the anchor symmetry) x
        # morpholine: the C-O cut bonds create both direct-only and
        # flipped-only candidates.  The flipped-only match must swap the
        # piece's dummy labels -- the two possible assemblies are
        # DIFFERENT molecules (O placement flips), and only the
        # homologous one ever appears.  Probe on rdkit 2026.09.1.
        from pharmaml.generative.scaffold_hop import (
            _excise_ring,
            _open_ring,
            _ring_units,
            _zip_child,
        )

        donor_smi, recipient_smi = "COC1CCOCC1CC", "C1COCC1"
        donor = rdkit.Chem.MolFromSmiles(donor_smi)
        recipient = rdkit.Chem.MolFromSmiles(recipient_smi)
        sig_r = {
            a.GetIdx(): (
                a.GetAtomicNum(), a.GetDegree(), a.GetIsAromatic(),
                a.GetFormalCharge(),
            )
            for a in recipient.GetAtoms()
        }
        ring_bonds = [
            b for b in recipient.GetBonds()
            if b.IsInRing() and b.GetBondType() == rdkit.Chem.BondType.SINGLE
        ]
        correct, wrong = set(), set()
        for ring, a1, a2, host_sigs in _ring_units(donor):
            for bond in ring_bonds:
                r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                direct = (
                    host_sigs[0] == sig_r[r1] and host_sigs[1] == sig_r[r2]
                )
                flipped = (
                    host_sigs[0] == sig_r[r2] and host_sigs[1] == sig_r[r1]
                )
                if not (direct or flipped):
                    continue
                opened = _open_ring(recipient, bond.GetIdx())
                excision = _excise_ring(donor, ring, a1, a2)
                assert opened is not None and excision is not None
                piece, _ = excision
                homologous = rdkit.Chem.Mol(piece)
                if not direct:  # the operator's swap: labels 1 <-> 2
                    for dummy in homologous.GetAtoms():
                        if dummy.GetAtomicNum() == 0:
                            dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
                child = _zip_child(opened, homologous, 120)
                if child is not None:
                    correct.add(child)
                crossed = rdkit.Chem.Mol(piece)
                if direct:  # the anti-swap: the non-homologous assembly
                    for dummy in crossed.GetAtoms():
                        if dummy.GetAtomicNum() == 0:
                            dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
                    child = _zip_child(opened, crossed, 120)
                    if child is not None:
                        wrong.add(child)
        assert correct == {"C1CCC2COCCC2OC1"}  # O neighbours match hosts
        assert wrong == {"C1CCC2CCOCC2OC1"}  # different constitution
        assert not correct & wrong
        for seed in range(80):
            child = scaffold_hop_crossover(
                donor_smi, recipient_smi, random.Random(seed)
            )
            if child is None:
                continue
            mol = rdkit.Chem.MolFromSmiles(child)
            assert mol is not None
            if len(mol.GetRingInfo().AtomRings()) == 2:  # a fuse child
                assert child in correct, (seed, child)

    def test_fuse_works_in_both_argument_orders(self) -> None:
        # ring parent first or second: the fuse scan covers both sides
        first = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "C1CCNCC1", random.Random(0)
        )
        second = scaffold_hop_crossover(
            "C1CCNCC1", "C1CCC2CCCCC2C1", random.Random(0)
        )
        assert first is not None and second is not None
        assert self._ring_sizes(first) == [6, 8]
        assert self._ring_sizes(second) == [6, 8]

    def test_ga_crossover_reaches_fuse_children(self) -> None:
        # the GA's _crossover wires the fuse geometry through: seed 0
        # draws a whole-ring splice (decalin x piperidine -> azadecalin)
        optimizer = GeneticOptimizer(QedObjective(), crossover="scaffold_hop")
        child = optimizer._crossover(
            "C1CCC2CCCCC2C1", "C1CCNCC1", random.Random(0)
        )
        assert child == "C1CCC2CCNCCCC2C1"
        assert self._ring_sizes(child) == [6, 8]

    def test_ga_hop_mode_absorbs_fused_donors_deterministically(self) -> None:
        # corpus with a fused-ring donor + a saturated acceptor: the
        # structural GA runs (the fuse geometry reachable from the
        # steroids-and-rings corpus) and stays a pure function of the seed
        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(), population_size=16, generations=3, seed=11,
                crossover="scaffold_hop",
            ).optimize(
                seed_smiles=["C1CCNCC1"],
                corpus=["C1CCNCC1", "C1CCC2CCCCC2C1", "C1Cc2ccccc2C1"],
            )

        first, second = run(), run()
        assert first.smiles == second.smiles
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)


class TestScaffoldHopRingInserts:
    """Ring-as-linker inserts (round 100x-31): the matrix's last cell.

    {excised piece} x {cut type} is now complete: acyclic pieces into
    ring cuts (transplant / transplant_path), ring pieces into ring
    cuts (fuse), acyclic pieces into acyclic cuts (bridge -- the
    fragments themselves swap), and ring pieces into ACYCLIC cuts --
    the **insert** geometry: a whole ring is excised from one parent
    (the same two-anchor donor machinery as fuse) and spliced INTO a
    cut acyclic bond of the other.  Both bridge halves are re-joined
    THROUGH the ring, so the ring becomes the spacer: ring-as-linker
    (decalin x hexane -> a dipropylcyclohexane, indane x heptane -> a
    dialkylbenzene).  The donor ring's former hosts must match the cut
    bond's endpoints (homologous flanks); a flipped-only match swaps
    the piece's dummy labels -- and the two possible assemblies are
    DIFFERENT molecules when the donor ring is asymmetric, so the flip
    is observable, exactly like the fuse geometry's.
    """

    @staticmethod
    def _ring_sizes(smiles: str) -> list[int]:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        assert mol is not None
        return sorted(len(ring) for ring in mol.GetRingInfo().AtomRings())

    def test_insert_splices_ring_into_acyclic_bond(self) -> None:
        # decalin x hexane, seed 5: the drawn candidate is an insert --
        # decalin's left 6-ring excised and spliced INTO a cut hexane
        # bond -> a single dipropylcyclohexane spacer ring, 12 atoms
        # (both parents' whole heavy-atom inventory), one fragment, no
        # dummy leaked.  Probe on rdkit 2026.09.1.
        child = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "CCCCCC", random.Random(5)
        )
        assert child == "CCCC1CCCCC1CCC"
        assert self._ring_sizes(child) == [6]
        mol = rdkit.Chem.MolFromSmiles(child)
        assert mol is not None
        assert len(rdkit.Chem.GetMolFrags(mol)) == 1
        assert mol.GetNumAtoms() == 12  # hexane's 6 + the donor 6-ring
        assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C"}
        assert not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        again = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "CCCCCC", random.Random(5)
        )
        assert again == child  # pure function of (parents, rng state)

    def test_insert_aromatic_donor_ring_becomes_the_spacer(self) -> None:
        # indane x heptane, seed 0: the BENZENE ring is the excised
        # donor, spliced into a cut heptane bond -> a dialkylbenzene;
        # the child keeps exactly one aromatic 6-ring (the spacer).
        # Probe on rdkit 2026.09.1.
        child = scaffold_hop_crossover(
            "C1Cc2ccccc2C1", "CCCCCCC", random.Random(0)
        )
        assert child == "CCCCCc1ccccc1CC"
        assert self._ring_sizes(child) == [6]
        mol = rdkit.Chem.MolFromSmiles(child)
        assert mol is not None
        assert mol.GetNumAtoms() == 13  # heptane's 7 + the benzene's 6
        aromatic_rings = [
            ring for ring in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        ]
        assert len(aromatic_rings) == 1  # the donor benzene, intact
        again = scaffold_hop_crossover(
            "C1Cc2ccccc2C1", "CCCCCCC", random.Random(0)
        )
        assert again == child

    def test_insert_children_are_always_valid_connected_and_capped(self) -> None:
        # invariant sweep: whatever geometry a seed draws, a non-None
        # child is a parseable, single-fragment, dummy-free molecule
        # under the cap -- the insert geometry obeys the module contract
        for parent_a, parent_b in (
            ("C1CCC2CCCCC2C1", "CCCCCC"),
            ("C1Cc2ccccc2C1", "CCCCCCC"),
            ("C1CCC2CCCCC2C1", "CCOCC"),
        ):
            for seed in range(40):
                child = scaffold_hop_crossover(
                    parent_a, parent_b, random.Random(seed)
                )
                if child is None:
                    continue
                assert len(child) <= 120, (parent_a, seed, child)
                mol = rdkit.Chem.MolFromSmiles(child)
                assert mol is not None, (parent_a, seed, child)
                assert len(rdkit.Chem.GetMolFrags(mol)) == 1
                assert not any(
                    a.GetAtomicNum() == 0 for a in mol.GetAtoms()
                ), (parent_a, seed, child)
                assert mol.GetRingInfo().NumRings() >= 1

    def test_insert_works_in_both_argument_orders(self) -> None:
        # ring donor first or second: the insert scan covers both sides
        # (hexane x decalin -- the ring donor as parent_b)
        first = scaffold_hop_crossover(
            "C1CCC2CCCCC2C1", "CCCCCC", random.Random(5)
        )
        second = scaffold_hop_crossover(
            "CCCCCC", "C1CCC2CCCCC2C1", random.Random(5)
        )
        assert first is not None and second is not None
        assert self._ring_sizes(first) == [6]
        assert self._ring_sizes(second) == [6]
        assert rdkit.Chem.MolFromSmiles(second).GetNumAtoms() == 12

    def test_flipped_insert_candidate_assembles_homologously(self) -> None:
        # asymmetric donor (ring O breaks the anchor symmetry) x diethyl
        # ether: the C-O cut bonds create both direct-only and
        # flipped-only candidates.  The flipped-only match must swap the
        # piece's dummy labels -- the two possible assemblies are
        # DIFFERENT molecules (the ring O lands beside a different
        # fragment), and only the homologous one ever appears.
        # Probe on rdkit 2026.09.1.
        import rdkit.Chem

        from pharmaml.generative.scaffold_hop import (
            _cut_into_two,
            _excise_ring,
            _ring_units,
            _zip_spliced,
        )

        donor_smi, recipient_smi = "COC1CCOCC1CC", "CCOCC"
        donor = rdkit.Chem.MolFromSmiles(donor_smi)
        recipient = rdkit.Chem.MolFromSmiles(recipient_smi)
        sig_r = {
            a.GetIdx(): (
                a.GetAtomicNum(), a.GetDegree(), a.GetIsAromatic(),
                a.GetFormalCharge(),
            )
            for a in recipient.GetAtoms()
        }
        acyclic_bonds = [
            b for b in recipient.GetBonds()
            if not b.IsInRing() and b.GetBondType() == rdkit.Chem.BondType.SINGLE
        ]
        correct, wrong = set(), set()
        for ring, a1, a2, host_sigs in _ring_units(donor):
            for bond in acyclic_bonds:
                b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                direct = (
                    host_sigs[0] == sig_r[b1] and host_sigs[1] == sig_r[b2]
                )
                flipped = (
                    host_sigs[0] == sig_r[b2] and host_sigs[1] == sig_r[b1]
                )
                if not (direct or flipped):
                    continue
                cut = _cut_into_two(recipient, bond.GetIdx())
                excision = _excise_ring(donor, ring, a1, a2)
                assert cut is not None and excision is not None
                begin_frag, end_frag = cut
                piece, _ = excision
                end_c = rdkit.Chem.Mol(end_frag)
                for atom in end_c.GetAtoms():  # the operator's relabel
                    if atom.GetAtomicNum() == 0:
                        atom.SetAtomMapNum(2)
                homologous = rdkit.Chem.Mol(piece)
                if not direct:
                    for dummy in homologous.GetAtoms():
                        if dummy.GetAtomicNum() == 0:
                            dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
                child = _zip_spliced(begin_frag, homologous, end_c, 120)
                if child is not None:
                    correct.add(child)
                crossed = rdkit.Chem.Mol(piece)
                if direct:  # the anti-swap: non-homologous assembly
                    for dummy in crossed.GetAtoms():
                        if dummy.GetAtomicNum() == 0:
                            dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
                    child = _zip_spliced(begin_frag, crossed, end_c, 120)
                    if child is not None:
                        wrong.add(child)
        assert correct == {"CCOC1CCOCC1CC"}  # O neighbours match hosts
        assert wrong == {"CCOC1COCCC1CC"}  # different constitution
        assert not correct & wrong
        # the recipient is acyclic, so bridge children exist alongside
        # insert children -- the shape filter cannot isolate inserts.
        # The rigorous assertions: the NON-homologous assembly never
        # appears in ANY child, and the homologous one is reachable.
        seen_correct: set[str] = set()
        for seed in range(80):
            child = scaffold_hop_crossover(
                donor_smi, recipient_smi, random.Random(seed)
            )
            if child is None:
                continue
            assert child not in wrong, (seed, child)
            if child in correct:
                seen_correct.add(child)
        assert seen_correct == correct  # the homologous insert is drawn

    def test_ga_crossover_reaches_insert_children(self) -> None:
        # the GA's _crossover wires the insert geometry through: seed 5
        # draws a ring-as-linker splice (decalin x hexane -> spacer ring)
        optimizer = GeneticOptimizer(QedObjective(), crossover="scaffold_hop")
        child = optimizer._crossover(
            "C1CCC2CCCCC2C1", "CCCCCC", random.Random(5)
        )
        assert child == "CCCC1CCCCC1CCC"
        assert self._ring_sizes(child) == [6]

    def test_ga_hop_mode_absorbs_insert_donors_deterministically(self) -> None:
        # corpus with a ring donor + a chain acceptor: the structural GA
        # runs (the insert geometry reachable) and stays a pure function
        # of the seed
        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(), population_size=16, generations=3, seed=11,
                crossover="scaffold_hop",
            ).optimize(
                seed_smiles=["CCCCCC"],
                corpus=["CCCCCC", "C1CCC2CCCCC2C1", "C1Cc2ccccc2C1"],
            )

        first, second = run(), run()
        assert first.smiles == second.smiles
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)


# ---------------------------------------------------------------------------
# Segment-swap crossover (100x-35): the sixth geometry
# ---------------------------------------------------------------------------
class TestScaffoldHopSegmentSwaps:
    """Two-point segment-swap crossover (round 100x-35): the sixth geometry.

    The graph analogue of classic two-point GA recombination: each
    parent's bounded acyclic MIDDLE (a single atom or a contiguous path
    of up to four atoms, the transplant eligibility) can be excised from
    the flank parent and REPLACED by the piece parent's homologous
    middle -- the only size-conserving geometry (insert grows a
    recipient, swap exchanges in place) and the medchem linker /
    bioisostere exchange move: an ester O swapped for an amide N
    (aspirin x paracetamol -> a salicylamide child), an ether O for a
    thioether S.  Homology: the two middles' host signatures must match
    direct or flipped; a flipped-only match excises the REVERSED piece
    path (the transplant_path trick).
    """

    def test_swap_exchanges_linkers_between_parents(self) -> None:
        # the killer demo probed in review: aspirin's ester O and
        # paracetamol's amide N sit in homologous environments (both
        # flanked by a carbonyl C and an aryl C), so the swap crosses
        # them -- seed 0 lands the salicylamide child, seed 5 the
        # O-for-N twin.  Probe on rdkit 2026.09.1.
        child = scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(0))
        assert child == "CC(=O)Nc1ccccc1C(=O)O"  # salicylamide: neither parent
        twin = scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(5))
        assert twin == "CC(=O)Oc1ccc(O)cc1"  # the O-for-N exchange

    def test_swap_child_combines_flanks_and_middle_chemistry(self) -> None:
        # the salicylamide child is a real molecule (CAS 87-17-2) that
        # is neither parent, and its substructure inventory proves the
        # exchange semantics: it carries the AMIDE linkage (aspirin has
        # none -- its acetyl ester O was swapped out) while keeping
        # aspirin's carboxylic-acid flank (paracetamol has none).  The
        # piece's chemistry really landed inside the flank's arms.
        child_mol = rdkit.Chem.MolFromSmiles("CC(=O)Nc1ccccc1C(=O)O")
        aspirin_mol = rdkit.Chem.MolFromSmiles(ASPIRIN)
        paracetamol_mol = rdkit.Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
        amide = rdkit.Chem.MolFromSmarts("C(=O)N")
        acid = rdkit.Chem.MolFromSmarts("C(=O)[OX2H1]")
        assert "CC(=O)Nc1ccccc1C(=O)O" not in (ASPIRIN, "CC(=O)Nc1ccc(O)cc1")
        # the amide arrived from the piece parent (paracetamol)
        assert child_mol.HasSubstructMatch(amide)
        assert not aspirin_mol.HasSubstructMatch(amide)
        assert paracetamol_mol.HasSubstructMatch(amide)
        # the carboxylic-acid flank survived from the flank parent (aspirin)
        assert child_mol.HasSubstructMatch(acid)
        assert aspirin_mol.HasSubstructMatch(acid)
        assert not paracetamol_mol.HasSubstructMatch(acid)

    def test_swap_conserves_the_flank_inventory(self) -> None:
        # size conservation, the geometry's defining invariant: the
        # child's heavy-atom count = flank parent's - flank middle's +
        # piece middle's.  Aspirin (13) - ester O (1) + amide N (1) = 13.
        child = scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(0))
        assert child == "CC(=O)Nc1ccccc1C(=O)O"
        mol = rdkit.Chem.MolFromSmiles(child)
        assert mol is not None
        assert mol.GetNumAtoms() == 13

    def test_swap_children_are_always_valid_connected_and_capped(self) -> None:
        # 40-seed sweep: every child parses, sanitises, stays one
        # connected fragment, leaks no dummy, and respects the length
        # cap.  The bioisostere pair exercises single-atom middles; the
        # linker pair exercises multi-atom path middles.
        for parent_a, parent_b in (
            (ASPIRIN, "CC(=O)Nc1ccc(O)cc1"),
            ("CCOCCOCC", "CCSCCSCC"),  # oxyethylene vs thioethylene chains
            ("c1ccccc1OCCc1ccccc1", "c1ccccc1SCCc1ccccc1"),
        ):
            for seed in range(40):
                child = scaffold_hop_crossover(parent_a, parent_b, random.Random(seed))
                assert child is None or canonical_smiles(child) is not None
                if child is None:
                    continue
                mol = rdkit.Chem.MolFromSmiles(child)
                assert mol is not None, (parent_a, parent_b, seed, child)
                assert len(rdkit.Chem.GetMolFrags(mol)) == 1
                assert not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
                assert len(child) <= 120

    def test_swap_deterministic_given_the_rng_state(self) -> None:
        for seed in (0, 3, 7):
            first = scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(seed))
            second = scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(seed))
            assert first == second

    def test_swap_single_atom_middles_exchange_bioisosteres(self) -> None:
        # CCOCC x CCSCC: the parents are themselves O/S bioisosteres, so
        # the single-atom swap children are the parents exchanged --
        # some seed must reach the S-in-O-side child (and vice versa),
        # proving the O <-> S middle exchange is reachable, not merely
        # admitted by the scan.
        children = {
            scaffold_hop_crossover("CCOCC", "CCSCC", random.Random(seed))
            for seed in range(24)
        }
        assert "CCSCC" in children  # O swapped out for S
        assert "CCOCC" in children  # S swapped out for O

    def test_swap_identical_parents_yield_the_parent(self) -> None:
        # swapping a middle for its homologous twin in the SAME molecule
        # rebuilds the parent -- tolerated exactly like the bridge
        # geometry's parent-echo pairings (the GA's fitness handles it).
        children = {
            scaffold_hop_crossover("CCOCC", "CCOCC", random.Random(seed))
            for seed in range(24)
        }
        assert "CCOCC" in children

    def test_swap_ring_only_parents_have_no_middles(self) -> None:
        # ring-only parents carry no bounded acyclic middle (and no
        # acyclic bond for any other geometry): the established None.
        assert scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None

    def test_swap_flipped_candidate_assembles_homologously(self) -> None:
        # white-box, the flipped assembly path exercised directly through
        # the helpers (no rng): flank CCOCC's middle (C1, O2) against
        # piece CCSCC's middle (S2, C3) is a FLIPPED-only match (the
        # flanks' host signatures read (deg-1 C, deg-2 C) vs the piece's
        # (deg-2 C, deg-1 C)).  The reversed excision assembles the
        # signature-matched child CCSCC; the naive direct excision would
        # bond the deg-1 host to the S side and yield the off-center
        # CCCSC -- the flip is observable.  Probe on rdkit 2026.09.1.
        from pharmaml.generative.scaffold_hop import (
            _excise_linker_path,
            _excise_middle,
            _zip_spliced,
        )

        flank_mol = rdkit.Chem.MolFromSmiles("CCOCC")
        piece_mol = rdkit.Chem.MolFromSmiles("CCSCC")
        assert flank_mol is not None and piece_mol is not None
        flanks = _excise_middle(flank_mol, (1, 2))  # the C1-O2 middle
        assert flanks is not None
        # flipped (correct): excise the REVERSED piece path (C3, S)
        exc_flipped = _excise_linker_path(piece_mol, (3, 2))
        # direct (the non-homologous assembly the flip guards against)
        exc_direct = _excise_linker_path(piece_mol, (2, 3))
        assert exc_flipped is not None and exc_direct is not None
        child_flipped = _zip_spliced(flanks[0], exc_flipped[0], flanks[1], 120)
        child_direct = _zip_spliced(flanks[0], exc_direct[0], flanks[1], 120)
        assert child_flipped == "CCSCC"  # homologous: S centred, 2+2 carbons
        assert child_direct == "CCCSC"  # off-center: the seam mismatch made real
        assert child_flipped != child_direct

    def test_excise_middle_builds_zippable_flanks(self) -> None:
        # the helper contract: cutting aspirin's ester O out leaves
        # exactly two single-dummy fragments, map-1 and map-2, whose
        # dummy hosts are the former middle hosts (the acetyl carbon
        # and the aryl carbon).
        from pharmaml.generative.scaffold_hop import _excise_middle

        mol = rdkit.Chem.MolFromSmiles(ASPIRIN)
        assert mol is not None
        # the ester O: degree 2, the only acyclic eligible single atom
        units = [
            (atom.GetIdx(), atom.GetSymbol())
            for atom in mol.GetAtoms()
            if atom.GetDegree() == 2
            and not atom.IsInRing()
            and atom.GetAtomicNum() != 0
        ]
        assert units == [(3, "O")]  # atom index 3: the ester oxygen
        flanks = _excise_middle(mol, (3,))
        assert flanks is not None
        flank_first, flank_second = flanks
        for fragment, map_num in ((flank_first, 1), (flank_second, 2)):
            dummies = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
            assert len(dummies) == 1
            assert dummies[0].GetAtomMapNum() == map_num
            host = dummies[0].GetNeighbors()[0]
            assert host.GetAtomicNum() == 6  # both hosts are carbons
        # the discarded middle: exactly one heavy atom went missing
        assert (
            flank_first.GetNumAtoms() + flank_second.GetNumAtoms()
            == mol.GetNumAtoms() - 1 + 2  # + 2 dummies
        )
        # a degree-3 "middle" (the carbonyl C) is rejected outright
        assert _excise_middle(mol, (0, 1)) is None

    def test_middle_units_enumeration_is_pinned(self) -> None:
        # determinism contract: singles in atom-index order, then the
        # _linker_path_units paths (2-4 atoms) -- a pure function of
        # the molecule.  Diethylene glycol OCCOCCO: terminal O's are
        # degree 1, the chain C-C-O-C-C contributes its sublists.
        from pharmaml.generative.scaffold_hop import _middle_units

        mol = rdkit.Chem.MolFromSmiles("OCCOCCO")
        assert mol is not None
        units = _middle_units(mol)
        paths = [path for path, _ in units]
        # atom order: O0 C1 C2 O3 C4 C5 O6 -> eligible chain C1-C2-O3-C4-C5
        assert paths == [
            (1,), (2,), (3,), (4,), (5,),
            (1, 2), (1, 2, 3), (1, 2, 3, 4),
            (2, 3), (2, 3, 4), (2, 3, 4, 5),
            (3, 4), (3, 4, 5),
            (4, 5),
        ]

    def test_ga_crossover_reaches_swap_children(self) -> None:
        # the GA's _crossover wires the swap geometry through the merged
        # pool (zero genetic.py code change): the aspirin x paracetamol
        # pair, seed 0, draws the salicylamide swap child.
        optimizer = GeneticOptimizer(QedObjective(), crossover="scaffold_hop")
        child = optimizer._crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", random.Random(0))
        assert child == "CC(=O)Nc1ccccc1C(=O)O"

    def test_ga_hop_mode_absorbs_swap_donors_deterministically(self) -> None:
        # corpus with the bioisostere pair: the structural GA runs (the
        # swap geometry reachable) and stays a pure function of the seed
        # -- the swap candidates join the pool AFTER the five
        # established geometries', so every prior pinned GA trajectory
        # survived byte-identical (the 202-test green run is the witness).
        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(), population_size=16, generations=3, seed=11,
                crossover="scaffold_hop",
            ).optimize(
                seed_smiles=[ASPIRIN],
                corpus=[ASPIRIN, "CC(=O)Nc1ccc(O)cc1", "NC(=O)c1ccccc1"],
            )

        first, second = run(), run()
        assert first.smiles == second.smiles
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)

    def test_corpus_swap_census_pinned(self) -> None:
        # the 100x-35 census: 123 of the 244 corpus molecules carry a
        # viable bounded acyclic middle; across all ordered parent
        # pairs, 3,298 pairs admit 7,328 swap candidates in total
        # (probed; matches the review agent's independent prototype).
        # 100x-36 extends the pin to the chemistry census the docs now
        # carry: 295 distinct linker chemistries exchange, 1,738 of the
        # candidates are heteroatom bioisostere exchanges (the docs'
        # README/module-docstring claims cannot drift silently).
        from pharmaml.generative.scaffold_hop import _middle_units

        mols = {}
        for smi in DEFAULT_CORPUS:
            mol = rdkit.Chem.MolFromSmiles(smi)
            if mol is not None:
                mols[smi] = (mol, _middle_units(mol))

        def chem(mol: object, path: tuple[int, ...]) -> str:
            return "".join(
                mol.GetAtomWithIdx(i).GetSymbol() for i in path  # type: ignore[attr-defined]
            )

        exchanges: dict[tuple[str, str], int] = {}
        for s_a, (m_a, units_a) in mols.items():
            for s_b, (m_b, units_b) in mols.items():
                if s_a == s_b:
                    continue
                for fp, fs in units_a:
                    for pp, ps in units_b:
                        direct = fs == ps
                        flipped = fs[0] == ps[1] and fs[1] == ps[0]
                        if direct or flipped:
                            key = (chem(m_a, fp), chem(m_b, pp))
                            exchanges[key] = exchanges.get(key, 0) + 1
        assert sum(1 for u in mols.values() if u[1]) == 123
        assert sum(exchanges.values()) == 7328
        assert len(exchanges) == 295
        het = {"O", "N", "S"}
        bioisostere = sum(
            n
            for (flank_chem, piece_chem), n in exchanges.items()
            if set(flank_chem) & het and set(piece_chem) & het and flank_chem != piece_chem
        )
        assert bioisostere == 1738
        # the classic ester/amide class is present and symmetric
        assert exchanges.get(("O", "N")) == 153
        assert exchanges.get(("N", "O")) == 153
        # 100x-37: the notebook's census line now quotes the conservative
        # C->C and the N<->C classes too -- pin them so the full line
        # cannot drift silently
        assert exchanges.get(("C", "C")) == 1438
        assert exchanges.get(("N", "C")) == 368
        assert exchanges.get(("C", "N")) == 368


# ---------------------------------------------------------------------------
# Ring-topology mutations (100x-33)
# ---------------------------------------------------------------------------
class TestRingTopologyMutations:
    """Contracts of the single-parent ring-size move set.

    ``ring_topology_moves`` enumerates every distinct one-step mutant
    (contraction / expansion over saturated single-bond rings, fused and
    spiro atoms/bonds conservatively skipped, the 5..8 window enforced);
    ``ring_topology_mutate`` draws one with the caller's rng or returns
    ``None``; the GA's ``mutation="topology"`` mode wires the draw with
    the character-mutation fallback.
    """

    def test_moves_pinned_piperidine(self) -> None:
        # piperidine -> cyclopentane (contract the N), pyrrolidine
        # (contract a CH2), azepane (expand) -- the whole ring-size axis
        assert ring_topology_moves("C1CCNCC1") == [
            "C1CCCC1",
            "C1CCCNCC1",
            "C1CCNC1",
        ]

    def test_moves_pinned_symmetric_and_capped(self) -> None:
        assert ring_topology_moves("C1CCCCC1") == ["C1CCCC1", "C1CCCCCC1"]
        assert ring_topology_moves("N1CCCC1") == ["C1CCNCC1"]  # 5-ring: expand only
        # 8-ring: contract only -- expansion past the window is blocked
        assert ring_topology_moves("N1CCCCCCC1") == ["C1CCCCCC1", "C1CCCNCC1"]

    def test_moves_morpholine_heteroatom_contraction(self) -> None:
        # removing the O -> pyrrolidine, the N -> tetrahydrofuran, a CH2
        # -> the oxazolidine; expansion -> the 7-ring oxazepane
        assert ring_topology_moves("C1COCCN1") == [
            "C1CCNC1",
            "C1CCOC1",
            "C1CNCCOC1",
            "C1COCN1",
        ]

    def test_aromatic_and_unsaturated_rings_untouched(self) -> None:
        assert ring_topology_moves("c1ccccc1") == []  # aromatic: never touched
        assert ring_topology_moves("C1=CCCCC1") == []  # unsaturated: never touched

    def test_fused_systems_conservative(self) -> None:
        # decalin: the 8 non-fused carbons all contract to the same
        # hydrindane (symmetry dedup); non-shared bonds expand; the four
        # fused atoms and the two shared bonds never move
        assert ring_topology_moves("C1CCC2CCCCC2C1") == [
            "C1CCC2CCCC2C1",
            "C1CCC2CCCCC2CC1",
        ]

    def test_invalid_and_acyclic_input_is_empty(self) -> None:
        assert ring_topology_moves("not-a-smiles") == []
        assert ring_topology_moves("") == []
        assert ring_topology_moves("CCO") == []

    def test_ring_size_window_validated_and_respected(self) -> None:
        with pytest.raises(DataValidationError, match="ring-size window"):
            ring_topology_moves("C1CCCCC1", min_ring_size=2, max_ring_size=8)
        with pytest.raises(DataValidationError, match="ring-size window"):
            ring_topology_moves("C1CCCCC1", min_ring_size=5, max_ring_size=5)
        # min=4 lets cyclopentane contract to cyclobutane
        assert ring_topology_moves("C1CCCC1", min_ring_size=4) == [
            "C1CCC1",
            "C1CCCCC1",
        ]
        # max=6 blocks piperidine's azepane expansion, keeps contraction
        assert ring_topology_moves("C1CCNCC1", max_ring_size=6) == [
            "C1CCCC1",
            "C1CCNC1",
        ]

    def test_mutate_deterministic_draw(self) -> None:
        assert ring_topology_mutate("C1COCCN1", random.Random(0)) == "C1COCN1"
        assert ring_topology_mutate("C1COCCN1", random.Random(0)) == "C1COCN1"

    def test_mutate_none_contract(self) -> None:
        assert ring_topology_mutate("CCO", random.Random(0)) is None
        assert ring_topology_mutate("not-a-smiles", random.Random(0)) is None

    def test_mutate_length_cap_filters_then_rejects(self) -> None:
        # cap 7 keeps three of morpholine's four moves (the 9-char
        # oxazepane is filtered out -- the draw changes accordingly)
        assert (
            ring_topology_mutate("C1COCCN1", random.Random(0), max_smiles_length=7)
            == "C1CCOC1"
        )
        # cap 6 rejects every morpholine move
        assert (
            ring_topology_mutate("C1COCCN1", random.Random(0), max_smiles_length=6)
            is None
        )

    def test_mutate_length_validation(self) -> None:
        with pytest.raises(DataValidationError, match="max_smiles_length"):
            ring_topology_mutate("C1CCNCC1", random.Random(0), max_smiles_length=0)

    def test_ga_topology_mode_wiring_and_determinism(self) -> None:
        ring_seeds = [s for s in DEFAULT_CORPUS if ring_topology_moves(s)][:12]

        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(),
                population_size=24,
                generations=6,
                seed=7,
                mutation="topology",
            ).optimize(seed_smiles=ring_seeds)

        first, second = run(), run()
        assert first.metadata["mutation"] == "topology"
        assert first.smiles == second.smiles  # deterministic given the seed
        assert first.best_score > 0.0
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)

    def test_ga_mode_validation_rejects_unknown_mutation(self) -> None:
        with pytest.raises(DataValidationError, match="mutation must be"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=1, mutation="rings"
            )

    def test_corpus_move_census_pinned(self) -> None:
        # the 100x-33 census: 30 of the 244 corpus molecules carry viable
        # ring-size moves, 133 distinct one-step mutants in total
        reachable = sum(1 for s in DEFAULT_CORPUS if ring_topology_moves(s))
        total = sum(len(ring_topology_moves(s)) for s in DEFAULT_CORPUS)
        assert reachable == 30
        assert total == 133


# ---------------------------------------------------------------------------
# 100x-38: adaptive operator selection -- the seventh generative capability
# ---------------------------------------------------------------------------
def _sat_ring_sizes(mol) -> list[int]:
    """Sizes of the fully-saturated single-bond rings (the movable ones)."""
    sizes: list[int] = []
    for ring in mol.GetRingInfo().AtomRings():
        bonds = [
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
            for i in range(len(ring))
        ]
        if all(
            bond.GetBondType() == rdkit.Chem.BondType.SINGLE
            and not bond.GetIsAromatic()
            for bond in bonds
        ):
            sizes.append(len(ring))
    return sizes


class _RingSizeBandObjective(ScoringObjective):
    """Gaussian band on the mean size of the saturated rings."""

    name = "ring_size_band"

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__()
        self.centre = (lo + hi) / 2.0
        self.sigma = max((hi - lo) / 2.0, 1e-6)

    def score(self, smiles: str) -> float:
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        sizes = _sat_ring_sizes(mol)
        if not sizes:
            return 0.0  # nothing movable: a ring objective cannot be satisfied
        z = (sum(sizes) / len(sizes) - self.centre) / self.sigma
        return math.exp(-0.5 * z * z)


@pytest.fixture(scope="class")
def ring_ab_runs() -> dict[str, object]:
    """The A/B on the ring-rich objective: topology mode, same seed + corpus.

    Only the operator-selection mode differs.  Probed values (deterministic,
    twice in fresh processes): adaptive final mean 0.807869 vs uniform
    0.764867 -- adaptive also wins best (0.965583 vs 0.961310).
    """
    objective = MultiObjective(
        [
            _RingSizeBandObjective(6.0, 8.0),
            QedObjective(),
            MolecularWeightBandObjective(220.0, 380.0),
        ]
    )
    parents = [s for s in DEFAULT_CORPUS if ring_topology_moves(s)]
    runs: dict[str, object] = {}
    for mode in ("adaptive", "uniform"):
        runs[mode] = GeneticOptimizer(
            objective,
            population_size=16,
            generations=16,
            seed=42,
            mutation="topology",
            operator_selection=mode,
        ).optimize(corpus=parents)
    return runs


class TestAdaptiveOperatorSelection:
    """The credit-assignment + softmax-with-floor operator draws."""

    STAT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "best_offspring",
            "best_offspring_score",
            "credit",
            "draws",
            "final_probability",
            "successes",
        }
    )

    # -- pin safety: the default path never moves --------------------------
    def test_default_path_is_bit_identical(self) -> None:
        kwargs: dict[str, object] = {
            "population_size": 8,
            "generations": 3,
            "seed": 42,
            "mutation": "topology",
        }
        default = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:12]
        )
        explicit = GeneticOptimizer(
            QedObjective(), operator_selection="uniform", **kwargs
        ).optimize(corpus=DEFAULT_CORPUS[:12])
        assert default.smiles == explicit.smiles
        assert default.history == explicit.history

    def test_default_metadata_untouched(self) -> None:
        run = GeneticOptimizer(
            QedObjective(), population_size=8, generations=2, seed=42
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        assert "operator_selection" not in run.metadata
        assert "operator_stats" not in run.metadata
        assert "operator_trajectory" not in run.metadata

    # -- the adaptive mode itself -------------------------------------------
    def test_adaptive_is_deterministic(self) -> None:
        kwargs: dict[str, object] = {
            "population_size": 8,
            "generations": 3,
            "seed": 42,
            "operator_selection": "adaptive",
        }
        first = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:10]
        )
        second = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:10]
        )
        assert first.smiles == second.smiles
        assert first.history == second.history
        assert first.metadata["operator_stats"] == second.metadata["operator_stats"]

    def test_adaptive_changes_the_search(self) -> None:
        kwargs: dict[str, object] = {
            "population_size": 8,
            "generations": 4,
            "seed": 42,
        }
        adaptive = GeneticOptimizer(
            QedObjective(), operator_selection="adaptive", **kwargs
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        uniform = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:10]
        )
        assert adaptive.smiles != uniform.smiles or adaptive.history != uniform.history

    def test_adaptive_reports_stats_shape(self) -> None:
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        assert run.metadata["operator_selection"] == "adaptive"
        stats = run.metadata["operator_stats"]
        assert set(stats) == {"level1", "geometry", "mutation"}
        assert set(stats["level1"]) == {"clone", "crossover"}
        assert set(stats["mutation"]) == {"character"}  # uniform mutation mode
        for operators in stats.values():
            for entry in operators.values():
                assert set(entry) == self.STAT_KEYS
                assert entry["draws"] >= entry["successes"] >= 0
                assert math.isfinite(entry["credit"])
                assert 0.0 <= entry["final_probability"] <= 1.0
                assert math.isfinite(entry["best_offspring_score"])
        level1_total = sum(
            entry["final_probability"] for entry in stats["level1"].values()
        )
        # the 2-op level hits 1e-9 by complementary 4dp rounding luck;
        # the 7-op geometry level can sit at 1 +/- ~5e-4 (rounding-budget
        # test below pins the documented deviation)
        assert abs(level1_total - 1.0) < 1e-9

    def test_final_probability_rounding_sum_budget(self) -> None:
        # 100x-40-r1 contract gap: final_probability (and credit /
        # best_offspring_score) are rounded to 4 decimal places for
        # display, so a level's rounded probabilities can sum to 1 with
        # an error of up to k * 5e-4 -- consumers must not renormalise
        # at a tight tolerance (the unrounded selection probabilities
        # are exact to 1e-12; documented in optimize's Returns notes).
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=0,
            crossover="scaffold_hop",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        stats = run.metadata["operator_stats"]
        for operators in stats.values():
            k = len(operators)
            total = sum(entry["final_probability"] for entry in operators.values())
            assert abs(total - 1.0) <= k * 5e-5 + 1e-12  # the rounding budget
        # probed, deterministic: this run's 7-op geometry level rounds to
        # 1.0001 -- the level1-only 1e-9 assertion does NOT generalise
        geometry_total = sum(
            entry["final_probability"] for entry in stats["geometry"].values()
        )
        assert geometry_total == pytest.approx(1.0001)

    def test_draw_counts_reconcile_with_attempts(self) -> None:
        # Splice mode: the geometry trace is never None (the splice
        # branch of the recombination always reports the "splice"
        # operator), so geometry draws EQUAL the crossover draws here.
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        stats = run.metadata["operator_stats"]
        level1_draws = sum(entry["draws"] for entry in stats["level1"].values())
        family_draws = sum(entry["draws"] for entry in stats["mutation"].values())
        assert level1_draws == family_draws  # every attempt draws both levels
        geometry_draws = sum(entry["draws"] for entry in stats["geometry"].values())
        assert geometry_draws == stats["level1"]["crossover"]["draws"]

        # Scaffold-hop mode: a pair with NO candidate cut traces a None
        # geometry (the pair falls back to parent_a without a geometry
        # draw), so geometry draws can LEGITIMATELY fall short of the
        # crossover draws -- the equality above is splice-mode luck, not
        # a general invariant.  Probed on DEFAULT_CORPUS[:12]: geometry
        # 33 < crossover 35 (deterministic; 100x-40-r1 finding 5).
        hopped = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            crossover="scaffold_hop",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:12])
        hstats = hopped.metadata["operator_stats"]
        hlevel1 = sum(entry["draws"] for entry in hstats["level1"].values())
        hfamily = sum(entry["draws"] for entry in hstats["mutation"].values())
        assert hlevel1 == hfamily  # every attempt draws both levels
        hgeometry = sum(entry["draws"] for entry in hstats["geometry"].values())
        hcrossover = hstats["level1"]["crossover"]["draws"]
        assert hgeometry <= hcrossover  # the documented inequality
        assert hgeometry < hcrossover  # this corpus hits the no-candidate trace

    def test_adaptive_splice_mode_degenerates_cleanly(self) -> None:
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            crossover="splice",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        geometry = run.metadata["operator_stats"]["geometry"]
        assert set(geometry) == {"splice"}  # one operator: bookkeeping only
        assert geometry["splice"]["final_probability"] == 1.0

    def test_adaptive_scaffold_mutation_mode(self) -> None:
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            mutation="scaffold",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        assert run.metadata["mutation"] == "scaffold"
        assert set(run.metadata["operator_stats"]["mutation"]) == {
            "scaffold",
            "character",
        }

    def test_scaffold_splice_bypass_level1_never_drawn(self) -> None:
        # 100x-40-r1 contract gap: with mutation="scaffold" +
        # crossover="splice" the character splice is bypassed, so the
        # level-1 book is never drawn -- yet operator_stats still
        # reports it with 0/0 draws and the neutral 0.5/0.5
        # probabilities (documented in optimize's Returns notes; the
        # mutation-family book keeps drawing, the geometry book does
        # not -- probed draws: family 9, geometry 0).
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            mutation="scaffold",
            crossover="splice",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        stats = run.metadata["operator_stats"]
        level1 = stats["level1"]
        assert set(level1) == {"clone", "crossover"}  # present but undrawn
        for entry in level1.values():
            assert entry["draws"] == 0
            assert entry["successes"] == 0
            assert entry["credit"] == 0.0
            assert entry["final_probability"] == 0.5
            assert entry["best_offspring"] is None
        assert sum(e["draws"] for e in stats["mutation"].values()) > 0
        assert sum(e["draws"] for e in stats["geometry"].values()) == 0

    # -- the flagship evidence: the ring-rich A/B ---------------------------
    def test_adaptive_improves_ring_objective_mean(self, ring_ab_runs) -> None:
        adaptive = ring_ab_runs["adaptive"].history[-1]["mean_score"]
        uniform = ring_ab_runs["uniform"].history[-1]["mean_score"]
        # r296: the exact probed means are Linux-build determinism pins --
        # the Windows runner's FP trajectory shifts the GA to an adaptive
        # mean of 0.834 under the SAME seed (numpy's RNG bit stream is
        # cross-platform; the RDKit-backed objective arithmetic above it
        # is not, and the selection cascade amplifies the ULPs). The
        # exact values assert only where they were probed; the CONTRACT
        # -- each mode in its probed regime and adaptive's improvement
        # over uniform -- runs on every host the release gate executes on.
        if sys.platform == "linux":
            assert round(adaptive, 3) == 0.808  # probed, deterministic
            assert round(uniform, 3) == 0.765
        assert 0.75 <= adaptive <= 0.90
        assert 0.70 <= uniform <= 0.82
        assert adaptive > uniform

    def test_adaptive_discovers_topology_family(self, ring_ab_runs) -> None:
        family = ring_ab_runs["adaptive"].metadata["operator_stats"]["mutation"]
        assert set(family) == {"topology", "character"}
        # the GA learned in-run what the notebook A/B showed: topology moves
        # pay off on a ring objective (final probability 0.896 when probed)
        assert family["topology"]["final_probability"] > 0.5
        assert family["topology"]["successes"] > family["character"]["successes"]

    def test_adaptive_starves_failing_family(self) -> None:
        acyclic = [
            s for s in DEFAULT_CORPUS if not ring_topology_moves(s)
        ][:16]
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=8,
            seed=7,
            mutation="topology",
            operator_selection="adaptive",
        ).optimize(corpus=acyclic)
        family = run.metadata["operator_stats"]["mutation"]
        topology, character = family["topology"], family["character"]
        assert topology["successes"] <= 2  # ring moves almost always fail here
        assert topology["final_probability"] < character["final_probability"]
        assert math.isfinite(topology["credit"])  # starved, never NaN
        assert math.isfinite(character["credit"])

    # -- the credit book unit contract --------------------------------------
    def test_credit_book_empty_window_is_neutral(self) -> None:
        from pharmaml.generative.genetic import _OperatorBook

        book = _OperatorBook(operators=("a", "b"), window=4, temperature=0.1)
        assert book.credit("a") == 0.0  # empty window reads as neutral
        probabilities = book.probabilities()
        assert set(probabilities) == {"a", "b"}
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
        assert all(abs(p - 0.5) < 1e-9 for p in probabilities.values())
        # exactly one rng.random() per draw (cumsum inversion)
        rng = random.Random(0)
        reference = random.Random(0)
        book.draw(rng)
        reference.random()
        assert rng.getstate() == reference.getstate()

    # -- the geometry_weights surface ----------------------------------------
    def test_geometry_weights_biases_the_draw(self) -> None:
        from collections import Counter

        from pharmaml.generative.scaffold_hop import _scaffold_hop_core

        parent_a, parent_b = ASPIRIN, "CC(=O)Nc1ccccc1O"  # aspirin x paracetamol
        census: dict[str, Counter] = {}
        for label, weights in (
            ("none", None),
            ("swap", {"swap": 100.0}),
            ("bridge", {"bridge": 100.0}),
        ):
            counter: Counter[str] = Counter()
            for seed in range(200):
                _, geometry = _scaffold_hop_core(
                    parent_a, parent_b, random.Random(seed), geometry_weights=weights
                )
                counter[geometry] += 1
            census[label] = counter
        assert set(census["none"]) == {"swap", "bridge"}  # mixed without weights
        assert census["swap"] == Counter({"swap": 200})  # fully forced
        assert census["bridge"] == Counter({"bridge": 200})
        # None is byte-identical to omitting the kwarg (50 seeds)
        for seed in range(50):
            rng_plain = random.Random(seed)
            rng_none = random.Random(seed)
            assert _scaffold_hop_core(parent_a, parent_b, rng_plain) == (
                _scaffold_hop_core(
                    parent_a, parent_b, rng_none, geometry_weights=None
                )
            )

    def test_geometry_weights_absent_geometry_renormalizes(self) -> None:
        from pharmaml.generative.scaffold_hop import _scaffold_hop_core

        parent_a, parent_b = ASPIRIN, "CC(=O)Nc1ccccc1O"
        # mass on an ABSENT geometry only -> no mass on what is present
        with pytest.raises(ValueError, match="place no mass on the geometries"):
            _scaffold_hop_core(
                parent_a, parent_b, random.Random(0), geometry_weights={"fuse": 100.0}
            )
        # absent geometries in a mixed dict simply renormalize away
        _, geometry = _scaffold_hop_core(
            parent_a,
            parent_b,
            random.Random(0),
            geometry_weights={"swap": 1.0, "fuse": 100.0},
        )
        assert geometry == "swap"

    def test_geometry_weights_empty_mapping_raises_none_uniform(self) -> None:
        # 100x-40-r1 contract gap: {} is a weights mapping that places
        # no mass on anything (the no-mass ValueError), NOT "no weights"
        # -- None / omitted keeps the historical uniform draw (the
        # distinction is documented in the geometry_weights parameter
        # description).
        with pytest.raises(ValueError, match="place no mass"):
            scaffold_hop_crossover(
                ASPIRIN,
                "CC(=O)Nc1ccccc1O",
                random.Random(0),
                geometry_weights={},
            )
        assert (
            scaffold_hop_crossover(ASPIRIN, "CC(=O)Nc1ccccc1O", random.Random(0))
            is not None
        )

    def test_geometry_weights_validation(self) -> None:
        with pytest.raises(ValueError, match="finite non-negative"):
            scaffold_hop_crossover(
                ASPIRIN,
                "CC(=O)Nc1ccccc1O",
                random.Random(0),
                geometry_weights={"swap": -1.0},
            )
        with pytest.raises(ValueError, match="finite non-negative"):
            scaffold_hop_crossover(
                ASPIRIN,
                "CC(=O)Nc1ccccc1O",
                random.Random(0),
                geometry_weights={"swap": float("nan")},
            )

    def test_geometry_weights_huge_int_keeps_the_valueerror_contract(self) -> None:
        # 100x-40-r1 finding 2: math.isfinite(2**1024) raises OverflowError
        # itself -- the coercion keeps the documented ValueError contract.
        with pytest.raises(ValueError, match="finite non-negative"):
            scaffold_hop_crossover(
                ASPIRIN,
                "CC(=O)Nc1ccccc1O",
                random.Random(0),
                geometry_weights={"swap": 2**1024},
            )

    def test_draw_rejects_empty_present_with_valueerror(self) -> None:
        # 100x-40-r1 finding 1: an empty `present` hit the float-tail guard
        # (IndexError on operators[-1]) -- now a typed caller-bug error.
        from pharmaml.generative.genetic import _OperatorBook

        book = _OperatorBook(operators=("a", "b"), window=4, temperature=0.1)
        with pytest.raises(ValueError, match="no operator of this book"):
            book.draw(random.Random(0), present=["foreign"])
        with pytest.raises(ValueError, match="no operator of this book"):
            book.draw(random.Random(0), present=[])

    # -- constructor validation + the wrapper + the CLI ---------------------
    def test_operator_selection_validation(self) -> None:
        with pytest.raises(DataValidationError, match="operator_selection must be"):
            GeneticOptimizer(
                QedObjective(),
                population_size=8,
                generations=2,
                operator_selection="nope",
            )
        with pytest.raises(DataValidationError, match="aos_window must be"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=2, aos_window=0
            )
        with pytest.raises(DataValidationError, match="aos_temperature must be"):
            GeneticOptimizer(
                QedObjective(),
                population_size=8,
                generations=2,
                aos_temperature=0.0,
            )
        with pytest.raises(DataValidationError, match="aos_temperature must be"):
            GeneticOptimizer(
                QedObjective(),
                population_size=8,
                generations=2,
                aos_temperature=float("nan"),
            )
        optimizer = GeneticOptimizer(QedObjective(), population_size=8, generations=2)
        assert optimizer.operator_selection == "uniform"
        assert optimizer.aos_window == 20
        assert optimizer.aos_temperature == pytest.approx(0.1)

    def test_run_genetic_optimization_passes_operator_selection_through(self) -> None:
        run = run_genetic_optimization(
            QedObjective(),
            population_size=8,
            generations=2,
            seed=42,
            operator_selection="adaptive",
            corpus=list(DEFAULT_CORPUS[:10]),
        )
        assert run.metadata["operator_selection"] == "adaptive"
        assert "operator_stats" in run.metadata

    def test_cli_adaptive_flag(self, capsys) -> None:
        from pharmaml.cli import main

        code = main(
            [
                "generative",
                "run",
                "--method",
                "ga",
                "--recipe",
                "qed",
                "--iterations",
                "3",
                "--population",
                "8",
                "--seed",
                "42",
                "--adaptive-operators",
            ]
        )
        assert code == 0
        output = capsys.readouterr().out
        assert "adaptive : on" in output
        assert "genetic_algorithm" in output

    def test_cli_adaptive_flag_rejects_rl(self, capsys) -> None:
        from pharmaml.cli import main

        code = main(
            [
                "generative",
                "run",
                "--method",
                "rl",
                "--recipe",
                "qed",
                "--iterations",
                "2",
                "--adaptive-operators",
            ]
        )
        assert code == 1  # the typed guard surfaces as a clean CLI error
        message = capsys.readouterr().err
        assert "--adaptive-operators selects the GA's adaptive operator" in message


class TestOperatorTrajectory:
    """The per-generation operator mix -- the "GA that learns" learning curve.

    ``metadata["operator_trajectory"]`` (adaptive runs only): one record
    per generation, index-aligned with ``history``, holding the
    full-precision selection probabilities in force at that generation's
    start -- record 0 is the fair prior, the final record the closing mix
    (the last generation produces no offspring).
    """

    # -- pin safety: the uniform path never moves --------------------------
    def test_uniform_metadata_has_no_trajectory_key(self) -> None:
        run = GeneticOptimizer(
            QedObjective(), population_size=8, generations=2, seed=42
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        assert "operator_trajectory" not in run.metadata

    # -- the trajectory artifact itself -------------------------------------
    def test_trajectory_is_aligned_with_history(self) -> None:
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        trajectory = run.metadata["operator_trajectory"]
        assert len(trajectory) == len(run.history)
        assert len(trajectory) == run.metadata["iterations"]
        assert [record["generation"] for record in trajectory] == [
            entry["generation"] for entry in run.history
        ]
        for record in trajectory:
            assert set(record) == {"generation", "level1", "geometry", "mutation"}
            for level in ("level1", "geometry", "mutation"):
                # unrounded probabilities: exact to 1e-9 (the 4dp rounding
                # budget is an operator_stats-only display artifact)
                assert abs(sum(record[level].values()) - 1.0) < 1e-9
                assert all(
                    math.isfinite(p) and 0.0 <= p <= 1.0
                    for p in record[level].values()
                )

    def test_first_record_is_the_fair_prior_and_last_matches_stats(self) -> None:
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            mutation="topology",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        trajectory = run.metadata["operator_trajectory"]
        # record 0: empty credit windows -- the exactly-fair prior (probed;
        # probabilities() on an empty book is the flat 0.5/0.5 mix)
        assert trajectory[0]["mutation"] == {"topology": 0.5, "character": 0.5}
        # the final record IS the closing mix: the last generation performs
        # no offspring production, so it equals the snapshot's rounded
        # final_probability within the documented 4dp budget (<= 5e-5)
        stats = run.metadata["operator_stats"]
        for op in ("topology", "character"):
            assert (
                abs(trajectory[-1]["mutation"][op]
                    - stats["mutation"][op]["final_probability"])
                <= 5e-5
            )

    def test_trajectory_is_deterministic(self) -> None:
        kwargs: dict[str, object] = {
            "population_size": 8,
            "generations": 3,
            "seed": 42,
            "operator_selection": "adaptive",
        }
        first = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:10]
        )
        second = GeneticOptimizer(QedObjective(), **kwargs).optimize(
            corpus=DEFAULT_CORPUS[:10]
        )
        assert first.metadata["operator_trajectory"] == (
            second.metadata["operator_trajectory"]
        )

    def test_generations_equals_one_yields_single_prior_record(self) -> None:
        # the 100x-40-r1 corner: generations=1 scores the initial population
        # and never produces offspring, so the trajectory is exactly ONE
        # record -- the all-fair prior (probed verbatim; corpus/seed-free
        # because every credit window is empty).  The default (uniform)
        # mutation mode's single-operator family is also pinned here: every
        # record of every such run is the constant {"character": 1.0}.
        run = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=1,
            seed=42,
            mutation="topology",
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        trajectory = run.metadata["operator_trajectory"]
        assert len(trajectory) == 1
        assert len(run.history) == 1
        assert trajectory[0] == {
            "generation": 0,
            "level1": {"crossover": 0.5, "clone": 0.5},
            "geometry": {"splice": 1.0},
            "mutation": {"topology": 0.5, "character": 0.5},
        }
        default_mutation = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=3,
            seed=42,
            operator_selection="adaptive",
        ).optimize(corpus=DEFAULT_CORPUS[:10])
        records = default_mutation.metadata["operator_trajectory"]
        assert records
        assert all(record["mutation"] == {"character": 1.0} for record in records)


# ---------------------------------------------------------------------------
# Minimum-heavy-atom sanity gate (100x-53)
# ---------------------------------------------------------------------------
class TestMinimumHeavyAtomGate:
    """The 'OO' comic and its opt-in gate (round 100x-53).

    Fail-first evidence: aspirin x naproxen at certain draw states breeds
    ``'OO'`` (hydrogen peroxide) / ``'CC'`` (ethane) as bridge children --
    single-fragment, sanitisable, under the cap, i.e. contract-conformant
    but chemically comic (98 of the first 400 seeds).  The gate is
    OPT-IN (``min_heavy_atoms``, default 1 = off) so every historical
    GA trajectory stays byte-identical; the default path must never move.
    """

    PARENT_A = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    PARENT_B = "COc1ccc2cc(ccc2c1)C(C)C(=O)O"  # naproxen

    @staticmethod
    def _heavy(smiles: str) -> int:
        return sum(1 for ch in smiles if ch.isalpha())

    def test_oo_comic_reproduced_then_gated(self) -> None:
        # seed 25 breeds the hydrogen-peroxide child verbatim (the comic)
        assert (
            scaffold_hop_crossover(self.PARENT_A, self.PARENT_B, random.Random(25))
            == "OO"
        )
        # the gate rejects it: None (no other viable pairing) or a real
        # child with at least the requested heavy atoms
        gated = scaffold_hop_crossover(
            self.PARENT_A, self.PARENT_B, random.Random(25), min_heavy_atoms=3
        )
        assert gated is None or self._heavy(gated) >= 3

    def test_default_gate_is_a_noop(self) -> None:
        # min_heavy_atoms=1 is the historical off state: identical output
        # on every seed (the GA determinism pins depend on this)
        for seed in range(30):
            default = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_B, random.Random(seed)
            )
            explicit = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_B, random.Random(seed), min_heavy_atoms=1
            )
            assert default == explicit

    def test_gate_rejects_the_tiny_class_over_a_seed_range(self) -> None:
        gated_tiny = 0
        for seed in range(60):
            child = scaffold_hop_crossover(
                self.PARENT_A, self.PARENT_B, random.Random(seed), min_heavy_atoms=3
            )
            assert child is None or self._heavy(child) >= 3
            if child is not None and self._heavy(child) < 3:
                gated_tiny += 1
        assert gated_tiny == 0

    def test_zip_child_unit_gate(self) -> None:
        from pharmaml.generative.scaffold_hop import _zip_child

        rdkit = pytest.importorskip("rdkit")
        frag_x = rdkit.Chem.MolFromSmiles("[*:1]O")
        frag_y = rdkit.Chem.MolFromSmiles("O[*:1]")
        assert _zip_child(frag_x, frag_y, 120) == "OO"
        assert _zip_child(frag_x, frag_y, 120, min_heavy_atoms=3) is None

    def test_zip_spliced_unit_gate(self) -> None:
        from pharmaml.generative.scaffold_hop import _zip_spliced

        rdkit = pytest.importorskip("rdkit")
        frag_begin = rdkit.Chem.MolFromSmiles("[*:1]C")
        piece = rdkit.Chem.MolFromSmiles("[*:1]O[*:2]")
        frag_end = rdkit.Chem.MolFromSmiles("C[*:2]")
        assert _zip_spliced(frag_begin, piece, frag_end, 120) == "COC"
        assert _zip_spliced(frag_begin, piece, frag_end, 120, min_heavy_atoms=4) is None

    def test_min_heavy_atoms_validation(self) -> None:
        for bad in (0, -3, True, "3", 2.5):
            with pytest.raises(ValueError, match="min_heavy_atoms"):
                scaffold_hop_crossover(
                    self.PARENT_A,
                    self.PARENT_B,
                    random.Random(0),
                    min_heavy_atoms=bad,
                )

    def test_numpy_integer_accepted(self) -> None:
        np = pytest.importorskip("numpy")
        child = scaffold_hop_crossover(
            self.PARENT_A, self.PARENT_B, random.Random(25), min_heavy_atoms=np.int64(3)
        )
        assert child is None or self._heavy(child) >= 3

    def test_optimizer_min_heavy_atoms_validation(self) -> None:
        for bad in (0, -1, True, "3"):
            with pytest.raises(DataValidationError, match="min_heavy_atoms"):
                GeneticOptimizer(QedObjective(), min_heavy_atoms=bad)

    def test_optimizer_threads_gate_end_to_end(self) -> None:
        # A length-minimising objective makes the GA keep the tiniest
        # breedable children; the gate must starve the <=2-heavy class
        # out of the crossover supply (mutation-deletion cascades can
        # still mint singles -- documented scope: the gate polices the
        # crossover, not the character-mutation fallback).
        class _TinyReward:
            def score(self, smiles: str) -> float:
                return -float(len(smiles))

        config = {
            "population_size": 32,
            "generations": 15,
            "crossover": "scaffold_hop",
            "mutation": "uniform",
            "mutation_rate": 0.01,
            "crossover_rate": 0.99,
            "seed": 7,
        }
        default = GeneticOptimizer(_TinyReward(), **config).optimize()
        gated = GeneticOptimizer(_TinyReward(), min_heavy_atoms=3, **config).optimize()
        default_tiny = sum(
            1 for s in default.to_dataframe()["smiles"] if self._heavy(s) <= 2
        )
        gated_tiny = sum(
            1 for s in gated.to_dataframe()["smiles"] if self._heavy(s) <= 2
        )
        # default run DOES converge to the comic class (seed-pinned)
        assert default_tiny >= 5
        # the gated run starves it (mutation cascades may leave a residue,
        # but far below the un-gated ecosystem)
        assert gated_tiny < default_tiny / 2


# ---------------------------------------------------------------------------
# Pareto front extraction (100x-56) — the multi-objective trade-off report
# ---------------------------------------------------------------------------
class TestParetoFront:
    """The non-dominated set of a MultiObjective run, pure and deterministic.

    Fail-first (100x-56): ``pharmaml.generative.pareto`` did not exist --
    ``from pharmaml.generative import pareto_front`` raised ImportError
    (/tmp/task56/probe_pareto.py), and the probe's GA run (population 24,
    8 generations, seed 11) showed WHY the report matters: 23 of 159
    molecules non-dominated, front members as deep as leaderboard rank
    154 -- trade-offs the weighted-sum ranking hides.
    """

    @staticmethod
    def _demo_result() -> GenerationResult:
        """A/B/D trade off; C is dominated by B (0.4>=0.3, 0.9>0.3)."""
        return GenerationResult(
            smiles=["A", "B", "C", "D"], scores=[0.9, 0.8, 0.7, 0.1],
            breakdown=[{"qed": 0.9, "novelty": 0.2},
                       {"qed": 0.4, "novelty": 0.9},
                       {"qed": 0.3, "novelty": 0.3},
                       {"qed": 0.85, "novelty": 0.25}],
        )

    # -- hand-built contract ------------------------------------------- #

    def test_demo_front_membership_and_dominance_counts(self) -> None:
        front = pareto_front(self._demo_result())
        assert isinstance(front, ParetoFront)
        assert front.objectives == ("qed", "novelty")  # first-appearance order
        assert front.smiles == ("A", "B", "D")  # leaderboard order preserved
        assert front.dominates == (0, 1, 0)  # B dominates C alone
        assert (front.n_total, front.n_excluded) == (4, 0)
        assert len(front) == 3 and not front.is_empty()

    def test_rank_one_is_always_on_the_front(self) -> None:
        # anything dominating rank 1 would outrank it under the weighted
        # geometric mean -- a structural invariant, checked over a sweep
        rng = random.Random(3)
        for _ in range(25):
            rows = [
                {"a": rng.random(), "b": rng.random(), "c": rng.random()}
                for _ in range(12)
            ]
            weights = [rng.random() + 0.1 for _ in rows[0]]
            totals = [
                (r["a"] ** weights[0] * r["b"] ** weights[1]
                 * r["c"] ** weights[2]) ** (1.0 / sum(weights))
                for r in rows
            ]
            order = sorted(range(len(rows)), key=lambda i: -totals[i])
            result = GenerationResult(
                smiles=[f"m{i}" for i in order],
                scores=[totals[i] for i in order],
                breakdown=[rows[i] for i in order],
            )
            front = pareto_front(result)
            assert result.smiles[0] in front.smiles

    def test_projection_subset_changes_the_front(self) -> None:
        result = GenerationResult(
            smiles=["E", "F", "G"], scores=[0.9, 0.5, 0.1],
            breakdown=[{"x": 0.9, "y": 0.1, "z": 0.1},
                       {"x": 0.2, "y": 0.9, "z": 0.9},
                       {"x": 0.5, "y": 0.05, "z": 0.5}],
        )
        # nobody dominates anybody on all three axes (E wins x, F wins y+z
        # partially, G is rescued by x against F and by z against E)
        assert set(pareto_front(result).smiles) == {"E", "F", "G"}
        # project onto (y, z): E (0.1, 0.1) and G (0.05, 0.5) both fall to
        # F (0.9, 0.9) -- the 2-objective front is strictly smaller
        assert pareto_front(result, objectives=["y", "z"]).smiles == ("F",)

    def test_identical_value_molecules_both_stay_on_the_front(self) -> None:
        result = GenerationResult(
            smiles=["P", "Q"], scores=[0.5, 0.5],
            breakdown=[{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}],
        )
        front = pareto_front(result)
        assert front.smiles == ("P", "Q")  # equal values: dominance needs a strict edge
        assert front.dominates == (0, 0)

    def test_nonfinite_rows_are_excluded_and_counted(self) -> None:
        result = GenerationResult(
            smiles=["A", "B", "N", "I"],
            scores=[0.9, 0.8, 0.5, 0.4],
            breakdown=[{"qed": 0.9, "novelty": 0.2},
                       {"qed": 0.4, "novelty": 0.9},
                       {"qed": float("nan"), "novelty": 0.7},
                       {"qed": 0.3, "novelty": float("inf")}],
        )
        front = pareto_front(result)
        assert front.n_total == 2  # only A and B entered the scan
        assert front.n_excluded == 2  # NaN and +inf rows dropped
        assert front.smiles == ("A", "B")
        assert "2 excluded" in str(front)

    # -- validation ------------------------------------------------------ #

    def test_single_objective_run_rejected_with_hint(self) -> None:
        result = GenerationResult(smiles=["CCO"], scores=[0.5])
        with pytest.raises(DataValidationError, match="no objective breakdown"):
            pareto_front(result)

    def test_unknown_objective_names_list_the_available(self) -> None:
        with pytest.raises(DataValidationError, match="available: \\['qed', 'novelty'\\]"):
            pareto_front(self._demo_result(), objectives=["qed", "weight"])

    def test_duplicate_objectives_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="duplicate"):
            pareto_front(self._demo_result(), objectives=["qed", "qed"])

    def test_two_objectives_required(self) -> None:
        with pytest.raises(DataValidationError, match="at least two objectives"):
            pareto_front(self._demo_result(), objectives=["qed"])

    # -- integration with a real MultiObjective GA run ------------------- #

    def test_ga_run_front_invariants_and_determinism(self) -> None:
        objective = MultiObjective(
            [QedObjective(), NoveltyObjective(), LogPBandObjective(1.0, 3.0)])
        config: ClassVar[dict[str, object]] = {
            "population_size": 24, "generations": 8, "seed": 11,
        }
        result = GeneticOptimizer(objective, **config).optimize()  # type: ignore[arg-type]
        front = pareto_front(result)
        # the probe's pinned shape: 23 of 159 non-dominated (seed 11)
        assert (front.n_total, len(front)) == (159, 23)
        assert front.objectives == ("qed", "novelty", "logp_band")
        # rank 1 structural invariant on a real run
        assert result.smiles[0] in front.smiles
        # every member is genuinely non-dominated (re-verified independently)
        rows = result.breakdown
        keys = list(front.objectives)

        def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
            return (
                all(a[k] >= b[k] for k in keys)
                and any(a[k] > b[k] for k in keys)
            )

        for member in front.smiles:
            values = rows[result.smiles.index(member)]
            assert not any(
                _dominates(other, values)
                for other in rows
                if other is not values
            )
        # determinism: a re-run reproduces the identical front
        again = GeneticOptimizer(objective, **config).optimize()  # type: ignore[arg-type]
        assert pareto_front(again).smiles == front.smiles

    def test_to_dataframe_and_str_audit(self) -> None:
        front = pareto_front(self._demo_result())
        frame = front.to_dataframe()
        assert list(frame.columns) == [
            "rank", "smiles", "qed", "novelty", "dominates", "crowding",
        ]
        assert frame["rank"].tolist() == [1, 2, 3]
        assert frame["smiles"].tolist() == ["A", "B", "D"]
        assert frame["dominates"].tolist() == [0, 1, 0]
        assert frame["crowding"].tolist() == [math.inf, math.inf, 2.0]
        text = str(front)
        assert text.startswith("Pareto front over 2 objectives [qed, novelty]: 3 of 4")
        assert "non-dominated" in text and "0 excluded" in text
        assert "pruned" not in text  # an un-pruned front never claims pruning

    def test_empty_front_str_audit(self) -> None:
        # every molecule dominated by the next -> a single-member front,
        # then check the empty-front branch with a synthetic container
        result = GenerationResult(
            smiles=["X", "Y"], scores=[1.0, 0.5],
            breakdown=[{"a": 1.0, "b": 1.0}, {"a": 0.5, "b": 0.5}],
        )
        front = pareto_front(result)
        assert front.smiles == ("X",)
        assert str(front).startswith("Pareto front over 2 objectives [a, b]: 1 of 2")


class TestParetoCrowding:
    """NSGA-II crowding distance + prune: the unwieldy-front shortlist.

    Fail-first context (100x-67): a real three-objective GA
    (population 300, 30 generations) leaves 46 of 8,108 molecules
    non-dominated -- a front too big to eyeball, and the CLI's old
    rank-truncation would have dropped the very corners the front
    exists to show.  Crowding distance ranks members by isolation;
    prune keeps the corners + the most isolated interiors.
    """

    @staticmethod
    def _five_member_result() -> GenerationResult:
        """Five front members + two dominated molecules (M6/M7).

        Hand-derived crowding over the front: qed spread 0.5, nov spread
        0.7; M3 = 0.6 + 0.5, M4 = 0.4 + 0.5, M5 = 0.7 + 0.5 (each interior
        gap normalised by its objective's range).  M1/M3 both dominate M7
        (0.8, 0.1), M4/M5 both dominate M6 (0.45, 0.5) -- so the front's
        dominates counts are (1, 0, 1, 1, 1).
        """
        return GenerationResult(
            smiles=["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
            scores=[0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5],
            breakdown=[{"qed": 0.9, "nov": 0.2},
                       {"qed": 0.4, "nov": 0.9},
                       {"qed": 0.85, "nov": 0.25},
                       {"qed": 0.5, "nov": 0.6},
                       {"qed": 0.6, "nov": 0.55},
                       {"qed": 0.45, "nov": 0.5},
                       {"qed": 0.8, "nov": 0.1}],
        )

    # -- crowding_distance contract ------------------------------------- #

    def test_crowding_boundaries_inf_interior_normalized_sum(self) -> None:
        front = pareto_front(self._five_member_result())
        # M1 best qed / worst nov, M2 worst qed / best nov -> both inf;
        # interiors: M3 1.1, M4 0.9, M5 1.2 (sum over both objectives)
        assert front.crowding_distance() == pytest.approx(
            (math.inf, math.inf, 1.1, 0.9, 1.2)
        )

    def test_crowding_two_member_front_both_boundary(self) -> None:
        result = GenerationResult(
            smiles=["P", "Q"], scores=[0.9, 0.5],
            breakdown=[{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}],
        )
        front = pareto_front(result)
        assert front.crowding_distance() == (math.inf, math.inf)

    def test_crowding_degenerate_objective_contributes_zero(self) -> None:
        # 'b' carries no spread (all 0.5) -> domination runs on 'a' alone;
        # the lone front member has nothing to measure -> 0.0, not inf
        result = GenerationResult(
            smiles=["P", "Q", "R"], scores=[0.9, 0.5, 0.1],
            breakdown=[{"a": 0.9, "b": 0.5},
                       {"a": 0.4, "b": 0.5},
                       {"a": 0.6, "b": 0.5}],
        )
        front = pareto_front(result)
        assert front.smiles == ("P",)  # P dominates Q and R on 'a' alone
        assert front.crowding_distance() == (0.0,)

    def test_crowding_identical_members_score_zero(self) -> None:
        result = GenerationResult(
            smiles=["P", "Q"], scores=[0.5, 0.5],
            breakdown=[{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}],
        )
        front = pareto_front(result)
        # equal values: dominance needs a strict edge -> both on the front,
        # no spread on either objective -> nothing to measure
        assert front.crowding_distance() == (0.0, 0.0)

    def test_crowding_empty_front_returns_empty_tuple(self) -> None:
        empty = ParetoFront(
            objectives=("a", "b"), smiles=(), values=(), dominates=(),
            n_total=0, n_excluded=0,
        )
        assert empty.crowding_distance() == ()
        assert empty.prune(3) is empty  # nothing to drop, identity

    def test_crowding_is_pure_and_deterministic(self) -> None:
        front = pareto_front(self._five_member_result())
        assert front.crowding_distance() == front.crowding_distance()

    # -- prune contract -------------------------------------------------- #

    def test_prune_keeps_corners_plus_most_isolated_interior(self) -> None:
        front = pareto_front(self._five_member_result())
        pruned = front.prune(3)
        # ranked: M1/M2 (inf, rank tie-break), then M5 (1.2) > M3 (1.1) >
        # M4 (0.9); leaderboard order re-emitted
        assert pruned.smiles == ("M1", "M2", "M5")

    def test_prune_carries_the_audit_fields(self) -> None:
        front = pareto_front(self._five_member_result())
        pruned = front.prune(2)
        assert pruned.n_front == 5            # the un-pruned size
        assert pruned.n_total == front.n_total
        assert pruned.n_excluded == front.n_excluded
        assert pruned.objectives == front.objectives
        # pairwise dominance cannot change with membership: the ORIGINAL
        # counts survive the cut (M1/M3 dominate M7, M4/M5 dominate M6)
        assert front.dominates == (1, 0, 1, 1, 1)
        assert pruned.dominates == (1, 0)

    def test_prune_str_is_honest_about_the_cut(self) -> None:
        front = pareto_front(self._five_member_result())
        text = str(front.prune(2))
        # the ORIGINAL front size leads (5 of 7 non-dominated); the prune
        # suffix carries the cut (2 of 5 kept)
        assert "5 of 7 molecules non-dominated" in text
        assert "pruned to the 2 most isolated of 5 by crowding distance" in text
        assert "strongest member 'M1' dominates 1 others" in text

    def test_prune_keep_at_least_size_returns_self(self) -> None:
        front = pareto_front(self._five_member_result())
        assert front.prune(5) is front       # exactly the size: no-op
        assert front.prune(99) is front      # beyond: no-op
        assert front.prune(5).n_front is None  # never claims pruning

    def test_prune_keep_below_one_rejected(self) -> None:
        front = pareto_front(self._five_member_result())
        with pytest.raises(DataValidationError, match="keep must be at least 1"):
            front.prune(0)
        with pytest.raises(DataValidationError, match="keep must be at least 1"):
            front.prune(-3)

    def test_prune_selection_is_monotonic_in_keep(self) -> None:
        # the ranked list is deterministic, so bigger keep extends the
        # selection -- never reshuffles it
        front = pareto_front(self._five_member_result())
        for smaller, larger in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            small = set(front.prune(smaller).smiles)
            large = set(front.prune(larger).smiles)
            assert small <= large

    def test_pruned_front_to_dataframe_keeps_conventions(self) -> None:
        front = pareto_front(self._five_member_result())
        frame = front.prune(3).to_dataframe()
        assert frame["smiles"].tolist() == ["M1", "M2", "M5"]
        assert frame["rank"].tolist() == [1, 2, 3]  # ranks re-dense
        # crowding is a property of the SET: the pruned front re-measures
        # isolation among its own members -- M5, interior between the two
        # corners on both objectives, now scores (0.9-0.4)/0.5 +
        # (0.9-0.2)/0.7 = 2.0 (it was 1.2 inside the five-member crowd)
        assert frame["crowding"].tolist() == [math.inf, math.inf, pytest.approx(2.0)]

    # -- integration with the pinned real GA run ------------------------- #

    def test_ga_run_prune_corners_and_monotonicity(self) -> None:
        objective = MultiObjective(
            [QedObjective(), NoveltyObjective(), LogPBandObjective(1.0, 3.0)])
        result = GeneticOptimizer(
            objective, population_size=24, generations=8, seed=11,
        ).optimize()
        front = pareto_front(result)
        assert (front.n_total, len(front)) == (159, 23)  # the pinned probe
        pruned = front.prune(5)
        # the selection is a subset of genuine front members
        assert set(pruned.smiles) <= set(front.smiles)
        assert len(pruned) == 5 and pruned.n_front == 23
        # monotonicity on a real front: prune(10) extends prune(5)
        assert set(pruned.smiles) <= set(front.prune(10).smiles)
        # every inf member of the original front survives a keep >= #inf
        distances = front.crowding_distance()
        inf_members = {
            front.smiles[i] for i, d in enumerate(distances) if math.isinf(d)
        }
        assert inf_members <= set(pruned.smiles)
        # determinism: a second prune reproduces the selection
        again = pareto_front(GeneticOptimizer(
            objective, population_size=24, generations=8, seed=11,
        ).optimize())
        assert again.prune(5).smiles == pruned.smiles


class TestBioisostereGaMode:
    """The ``mutation="bioisostere"`` GA mode (round 100x-152 wiring).

    The module shipped in an earlier round but was never wired into
    the package surface or the GA -- its own doctest imported
    ``from pharmaml.generative import bioisostere_moves`` and FAILED
    under a bare ``pytest`` (the audit's red-suite bomb).  Both wires
    are pinned here: the package-level export and the GA mode.
    """

    def test_package_level_exports(self) -> None:
        from pharmaml.generative import bioisostere_moves, bioisostere_mutate

        assert callable(bioisostere_moves)
        assert callable(bioisostere_mutate)

    def test_moves_benzene_to_pyridine(self) -> None:
        from pharmaml.generative import bioisostere_moves

        assert bioisostere_moves("c1ccccc1") == ["c1ccncc1"]

    def test_mutate_draw_is_seed_deterministic(self) -> None:
        import random

        from pharmaml.generative import bioisostere_mutate

        first = bioisostere_mutate("CC(=O)Oc1ccccc1C(=O)O", random.Random(42))
        second = bioisostere_mutate("CC(=O)Oc1ccccc1C(=O)O", random.Random(42))
        assert first == second
        assert first is not None

    def test_mutate_none_for_unparseable(self) -> None:
        import random

        from pharmaml.generative import bioisostere_mutate

        assert bioisostere_mutate("not-a-smiles", random.Random(0)) is None

    def test_ga_bioisostere_mode_wiring_and_determinism(self) -> None:
        from pharmaml.generative import GeneticOptimizer
        from pharmaml.generative.scoring import QedObjective

        hetero_seeds = [
            s for s in DEFAULT_CORPUS if any(ch in s for ch in ("N", "O", "S"))
        ][:12]

        def run() -> GenerationResult:
            return GeneticOptimizer(
                QedObjective(),
                population_size=24,
                generations=6,
                seed=7,
                mutation="bioisostere",
            ).optimize(seed_smiles=hetero_seeds)

        first, second = run(), run()
        assert first.metadata["mutation"] == "bioisostere"
        assert first.smiles == second.smiles  # deterministic given the seed
        assert first.best_score > 0.0
        assert all(canonical_smiles(smi) is not None for smi in first.smiles)

    def test_ga_bioisostere_mode_adaptive_operator_book(self) -> None:
        # adaptive operator selection must see the two-operator
        # mutation family (bioisostere + character), like topology's.
        from pharmaml.generative import GeneticOptimizer
        from pharmaml.generative.scoring import QedObjective

        optimizer = GeneticOptimizer(
            QedObjective(),
            population_size=8,
            generations=2,
            seed=3,
            mutation="bioisostere",
            operator_selection="adaptive",
        )
        result = optimizer.optimize(seed_smiles=["CCO", "CCN", "CCS"])
        mutation_stats = result.metadata.get("operator_stats", {}).get("mutation", {})
        assert set(mutation_stats) == {"bioisostere", "character"}

    def test_unknown_mutation_still_rejected(self) -> None:
        from pharmaml.generative import GeneticOptimizer
        from pharmaml.generative.scoring import QedObjective

        with pytest.raises(DataValidationError, match="mutation must be"):
            GeneticOptimizer(
                QedObjective(), population_size=8, generations=1, mutation="isosteres"
            )

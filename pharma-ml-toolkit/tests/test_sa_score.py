"""Synthetic Accessibility (SA) score tests -- MASTER_PROMPT part3 §24.

The Ertl-Schuffenhauer score (J Cheminform 1:8, 2009) as absorbed into
:mod:`pharmaml.chem.sa_scoring`: direction, range, reference values,
invalid-SMILES safety, the §24.3 results table, the §24.4 filter that
preserves rejected candidates, the §24.5 generation objective, and the
provenance/limitations copy that must ship with every score.

Reference values were pinned against the official RDKit Contrib
``sascorer.py`` (same wheel, same fragment database) on this exact
stack: aspirin 1.580, ethanol 1.980, caffeine 2.298, the taxane-style
complex 4.076.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pharmaml.chem.sa_scoring import (
    SA_SCORE_LIMITATIONS,
    SA_SCORE_METHOD,
    SAFilterResult,
    SAScoreBatch,
    SAScoreResult,
    filter_by_sa,
    sa_score,
    sa_score_frame,
    sa_score_summary,
    sa_scores,
)
from pharmaml.exceptions import DataValidationError
from pharmaml.generative import MultiObjective, QedObjective, SAScoreObjective

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
TAXANE = (
    "CC1=C(C[C@@H]2[C@]1(C[C@H]([C@@H](C2=O)O)NC(=O)c3ccccc3)O)OC(=O)C"
)


class TestSASingle:
    """§24.2 -- one molecule, full validation context."""

    def test_reference_aspirin(self) -> None:
        result = sa_score(ASPIRIN)
        assert result.valid
        assert result.score is not None
        assert abs(result.score - 1.58) < 0.02

    def test_reference_caffeine(self) -> None:
        assert abs(sa_score(CAFFEINE).score - 2.298) < 0.02

    def test_range_contract(self) -> None:
        for smiles in (ASPIRIN, ETHANOL, CAFFEINE, TAXANE):
            score = sa_score(smiles).score
            assert score is not None and 1.0 <= score <= 10.0

    def test_direction_lower_is_easier(self) -> None:
        """The §24.2 direction rule: harder molecules score higher."""
        assert sa_score(TAXANE).score > sa_score(CAFFEINE).score
        assert sa_score(CAFFEINE).score > sa_score(ASPIRIN).score

    def test_result_carries_direction_and_provenance(self) -> None:
        result = sa_score(ASPIRIN)
        assert "Ertl" in result.method
        assert "lower" in result.direction
        assert "easier" in result.direction

    def test_canonical_smiles_present(self) -> None:
        assert sa_score(ASPIRIN).canonical_smiles == ASPIRIN

    def test_invalid_smiles_is_safe(self) -> None:
        result = sa_score("not-a-smiles")
        assert not result.valid
        assert result.score is None
        assert result.error
        assert result.smiles == "not-a-smiles"  # preserved for debugging

    def test_empty_string_is_invalid(self) -> None:
        result = sa_score("   ")
        assert not result.valid
        assert result.score is None

    def test_mol_object_accepted(self) -> None:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ETHANOL)
        assert sa_score(mol).score == sa_score(ETHANOL).score

    def test_as_dict_is_json_friendly(self) -> None:
        row = sa_score(ASPIRIN).as_dict()
        assert row["valid"] is True
        assert row["canonicalSmiles"] == ASPIRIN
        assert "direction" in row and "method" in row


class TestSABatch:
    """§24.2/24.3 -- batch scoring with validity accounting."""

    def test_accounting(self) -> None:
        batch = sa_scores([ASPIRIN, "not-a-smiles", CAFFEINE])
        assert isinstance(batch, SAScoreBatch)
        assert batch.n_scored == 2
        assert batch.n_invalid == 1
        assert len(batch.results) == 3

    def test_sorted_by_score_easiest_first(self) -> None:
        batch = sa_scores([CAFFEINE, ASPIRIN, TAXANE])
        ranked = batch.sorted_by_score()
        assert [r.smiles for r in ranked] == [ASPIRIN, CAFFEINE, TAXANE]

    def test_sorted_descending(self) -> None:
        batch = sa_scores([CAFFEINE, ASPIRIN, TAXANE])
        ranked = batch.sorted_by_score(descending=True)
        assert ranked[0].smiles == TAXANE

    def test_empty_batch(self) -> None:
        batch = sa_scores([])
        assert batch.n_scored == 0
        assert batch.n_invalid == 0


class TestSAFrame:
    """§24.3 -- the generative-results table."""

    def test_frame_shape_and_columns(self) -> None:
        frame = sa_score_frame([ASPIRIN, "not-a-smiles"])
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == [
            "index", "smiles", "canonical_smiles", "sa_score", "valid", "error",
        ]
        assert len(frame) == 2
        assert frame["sa_score"].isna().tolist() == [False, True]

    def test_frame_sortable_by_score(self) -> None:
        frame = sa_score_frame([TAXANE, ASPIRIN, CAFFEINE])
        ordered = frame.sort_values("sa_score")
        assert ordered["smiles"].iloc[0] == ASPIRIN

    def test_frame_preserves_invalid_rows(self) -> None:
        frame = sa_score_frame(["bad-smiles"])
        assert not bool(frame["valid"].iloc[0])
        assert frame["error"].iloc[0]


class TestSAFilter:
    """§24.4 -- inclusive-range filtering that keeps the rejected."""

    def _results(self) -> list[SAScoreResult]:
        return list(sa_scores([ASPIRIN, CAFFEINE, TAXANE, "junk"]).results)

    def test_max_sa_threshold(self) -> None:
        outcome = filter_by_sa(self._results(), max_sa=2.5)
        assert isinstance(outcome, SAFilterResult)
        assert outcome.n_kept == 2          # aspirin + caffeine
        assert outcome.n_rejected == 2      # taxane + junk

    def test_min_sa_threshold(self) -> None:
        outcome = filter_by_sa(self._results(), min_sa=2.0)
        kept = [r.smiles for r in outcome.kept]
        assert CAFFEINE in kept and TAXANE in kept
        assert ASPIRIN not in kept

    def test_inclusive_bounds(self) -> None:
        caffeine = sa_score(CAFFEINE).score
        outcome = filter_by_sa(
            [sa_score(CAFFEINE)], min_sa=caffeine, max_sa=caffeine
        )
        assert outcome.n_kept == 1  # bounds are inclusive

    def test_no_filter_keeps_everything(self) -> None:
        outcome = filter_by_sa(self._results())
        assert outcome.n_kept == 4          # invalid rows included: no filter
        assert outcome.n_rejected == 0

    def test_invalid_rows_rejected_when_filter_active(self) -> None:
        outcome = filter_by_sa(self._results(), max_sa=2.5)
        assert all(r.valid for r in outcome.kept)
        assert any(not r.valid for r in outcome.rejected)

    def test_rejected_preserved_for_later_analysis(self) -> None:
        outcome = filter_by_sa(self._results(), max_sa=2.5)
        rejected_smiles = {r.smiles for r in outcome.rejected}
        assert TAXANE in rejected_smiles and "junk" in rejected_smiles

    def test_min_above_max_rejected(self) -> None:
        with pytest.raises(DataValidationError):
            filter_by_sa(self._results(), min_sa=5.0, max_sa=2.0)


class TestSASummary:
    """§24.3 -- batch statistics over actually computed values."""

    def test_summary_fields(self) -> None:
        summary = sa_score_summary([ASPIRIN, CAFFEINE, "junk"])
        assert summary["n_total"] == 3
        assert summary["n_scored"] == 2
        assert summary["n_invalid"] == 1
        assert summary["min"] <= summary["median"] <= summary["max"]
        assert "Ertl" in summary["method"]
        assert "limitations" in summary

    def test_limitations_documented(self) -> None:
        assert "not" in SA_SCORE_LIMITATIONS.lower()
        assert "route" in SA_SCORE_LIMITATIONS.lower()

    def test_all_invalid_summary(self) -> None:
        summary = sa_score_summary(["junk", "junk2"])
        assert summary["n_scored"] == 0
        assert "min" not in summary  # no fabricated statistics


class TestSAScoreObjective:
    """§24.5 -- SA as an optimisation objective inside the generative stack."""

    def test_reward_monotone_against_sa(self) -> None:
        obj = SAScoreObjective()
        reward_aspirin = obj.score(ASPIRIN)
        reward_caffeine = obj.score(CAFFEINE)
        reward_taxane = obj.score(TAXANE)
        assert reward_aspirin >= reward_caffeine >= reward_taxane

    def test_easy_molecule_earns_full_reward(self) -> None:
        assert SAScoreObjective().score(ASPIRIN) == 1.0

    def test_invalid_smiles_scores_zero_and_counts(self) -> None:
        obj = SAScoreObjective()
        assert obj.score("not-a-smiles") == 0.0
        assert obj.n_invalid == 1
        assert obj.n_scored == 1

    def test_custom_anchors(self) -> None:
        obj = SAScoreObjective(easy=2.0, hard=3.0)
        caffeine = sa_score(CAFFEINE).score
        expected = (3.0 - caffeine) / (3.0 - 2.0)
        assert abs(obj.score(CAFFEINE) - min(max(expected, 0.0), 1.0)) < 1e-9

    def test_anchor_validation(self) -> None:
        with pytest.raises(DataValidationError):
            SAScoreObjective(easy=5.0, hard=3.0)
        with pytest.raises(DataValidationError):
            SAScoreObjective(easy=0.0, hard=3.0)   # below the [1,10] scale
        with pytest.raises(DataValidationError):
            SAScoreObjective(easy=2.0, hard=12.0)  # above the scale

    def test_multi_objective_integration(self) -> None:
        """Inside MultiObjective a hard SA anchor vetoes, a fair score passes."""
        objective = MultiObjective(
            [QedObjective(), SAScoreObjective(easy=2.0, hard=2.5)]
        )
        aspirin = objective.score(ASPIRIN)     # SA 1.58 below easy -> no veto
        taxane = objective.score(TAXANE)       # SA ~4 -> SA reward 0 -> veto
        assert taxane == 0.0
        assert aspirin > 0.0

    def test_objective_name_registered(self) -> None:
        assert SAScoreObjective().name == "sa_score"


class TestProvenance:
    """The fragment model must be the RDKit-shipped database, not a copy."""

    def test_method_citation(self) -> None:
        assert "2009" in SA_SCORE_METHOD
        assert "RDKit" in SA_SCORE_METHOD

    def test_module_importable_without_scoring(self) -> None:
        """Importing never loads the fragment database (lazy gate).

        Checked in a fresh interpreter: earlier tests in this session
        have already warmed the cache by design.
        """
        import subprocess
        import sys

        code = (
            "import pharmaml.chem.sa_scoring as m, sys; "
            "sys.exit(0 if m._FRAGMENT_SCORES is None else 1)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,  # the exit code IS the contract
        )
        assert proc.returncode == 0, proc.stderr


def test_doctest_module() -> None:
    """The module docstring's worked examples stay true."""
    import doctest

    results = doctest.testmod(
        __import__("pharmaml.chem.sa_scoring", fromlist=["x"]),
        verbose=False,
    )
    assert results.attempted > 0
    assert results.failed == 0

"""Tests for :mod:`pharmaml.ml.sar` — MMP, activity cliffs, R-groups.

The fragmentation half is pure graph math (no RDKit) and is exercised on
hand-checked molecules; the R-group half runs against real RDKit SMARTS
matching and is skipped when RDKit is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.sar import (
    MATCHED_PAIR_COLUMNS,
    ActivityCliffReport,
    MatchedPair,
    RGroup,
    activity_cliffs,
    matched_pairs,
    mmp_deltas,
    mmp_fragmentation,
    rgroup_table,
)


# --------------------------------------------------------------------------- #
# mmp_fragmentation (pure graph)
# --------------------------------------------------------------------------- #
class TestMmpFragmentation:
    def test_ethanol_gives_both_directions(self):
        assert mmp_fragmentation("CCO") == [("CC", "O"), ("CO", "C")]

    def test_ring_only_molecule_has_no_acyclic_bond(self):
        assert mmp_fragmentation("c1ccccc1") == []

    def test_double_bond_is_not_cut(self):
        # acetic acid: only the C-O single bond is cut; the C=O survives whole
        pairs = mmp_fragmentation("CC(=O)O")
        assert ("CC=O", "O") in pairs
        # the carboxyl fragment is never split into "C" and "=O"
        assert all(sub not in {"C", "O", "=O"} or ctx != "C" for ctx, sub in pairs)

    def test_hostile_input_is_an_honest_empty(self):
        assert mmp_fragmentation("not a smiles at all") == []
        assert mmp_fragmentation("") == []

    def test_max_cuts_guard(self):
        with pytest.raises(DataValidationError, match="max_cuts=1"):
            mmp_fragmentation("CCO", max_cuts=2)

    def test_larger_fragment_is_the_context(self):
        # butane: every cut should leave the bigger piece as the context
        for context, sub in mmp_fragmentation("CCCC"):
            assert len(context) >= len(sub)

    def test_fragments_are_deduplicated(self):
        pairs = mmp_fragmentation("CCO")
        assert len(pairs) == len(set(pairs))


# --------------------------------------------------------------------------- #
# matched_pairs
# --------------------------------------------------------------------------- #
class TestMatchedPairs:
    def _frame(self):
        return pd.DataFrame(
            {"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"], "pic50": [5.0, 6.5, 5.1]}
        )

    def test_schema_and_ordering(self):
        pairs = matched_pairs(self._frame(), "smiles", "pic50")
        assert list(pairs.columns) == list(MATCHED_PAIR_COLUMNS)
        assert pairs["delta"].is_monotonic_decreasing

    def test_delta_is_nonnegative_and_lower_on_left(self):
        pairs = matched_pairs(self._frame(), "smiles", "pic50")
        assert (pairs["delta"] >= 0).all()
        assert (pairs["left_activity"] <= pairs["right_activity"]).all()

    def test_pair_count(self):
        # phenol/aniline/toluene share the benzene context -> 3 pairs
        pairs = matched_pairs(self._frame(), "smiles", "pic50")
        assert len(pairs) == 3

    def test_min_shared_filters_singleton_contexts(self):
        # a lone molecule cannot pair with itself
        frame = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "y": [1.0, 2.0]})
        assert matched_pairs(frame, "smiles", "y").empty
        assert matched_pairs(frame, "smiles", "y", min_shared=2).empty

    def test_missing_column_raises(self):
        with pytest.raises(DataValidationError, match="not in frame"):
            matched_pairs(self._frame(), "nope", "pic50")

    def test_non_numeric_activity_raises(self):
        frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": ["a", "b"]})
        with pytest.raises(DataValidationError, match="must be numeric"):
            matched_pairs(frame, "smiles", "y")

    def test_negative_min_shared_raises(self):
        with pytest.raises(DataValidationError, match="min_shared"):
            matched_pairs(self._frame(), "smiles", "pic50", min_shared=-1)

    def test_non_finite_activity_dropped(self):
        frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": [np.nan, 6.0]})
        assert matched_pairs(frame, "smiles", "y").empty

    def test_empty_result_keeps_schema(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1"], "y": [1.0]})
        pairs = matched_pairs(frame, "smiles", "y")
        assert list(pairs.columns) == list(MATCHED_PAIR_COLUMNS)


# --------------------------------------------------------------------------- #
# activity_cliffs
# --------------------------------------------------------------------------- #
class TestActivityCliffs:
    def test_threshold_filters(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.5]})
        report = activity_cliffs(frame, "smiles", "y", min_delta=1.0)
        assert isinstance(report, ActivityCliffReport)
        assert len(report) == 1
        assert round(float(report.pairs["delta"].iloc[0]), 3) == 1.5
        assert report.n_pairs_scanned == 1

    def test_high_threshold_yields_nothing(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.5]})
        report = activity_cliffs(frame, "smiles", "y", min_delta=10.0)
        assert len(report) == 0
        assert report.n_pairs_scanned == 1

    def test_negative_threshold_raises(self):
        frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": [5.0, 6.0]})
        with pytest.raises(DataValidationError, match="min_delta"):
            activity_cliffs(frame, "smiles", "y", min_delta=-1.0)

    def test_as_frame_roundtrip(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.5]})
        report = activity_cliffs(frame, "smiles", "y")
        assert report.as_frame() is report.pairs

    def test_summary_names_the_counts(self):
        frame = pd.DataFrame(
            {"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"], "y": [5.0, 6.5, 5.1]}
        )
        report = activity_cliffs(frame, "smiles", "y", min_delta=1.0)
        assert report.summary() == (
            "2 activity cliff(s) at |delta| >= 1, out of 3 matched pair(s)"
        )


# --------------------------------------------------------------------------- #
# mmp_deltas
# --------------------------------------------------------------------------- #
class TestMmpDeltas:
    def test_aggregation(self):
        # CCO/CCCO/CCOC: the O->CO swap recurs under the shared "CC" context
        frame = pd.DataFrame(
            {"smiles": ["CCO", "CCCO", "CCOC"], "y": [1.0, 2.0, 3.0]}
        )
        deltas = mmp_deltas(frame, "smiles", "y")
        assert "mean_delta" in deltas.columns
        o_to_co = deltas[(deltas["left_sub"] == "O") & (deltas["right_sub"] == "CO")]
        assert int(o_to_co["n_pairs"].iloc[0]) == 2
        assert round(float(o_to_co["mean_delta"].iloc[0]), 6) == 1.5

    def test_min_pairs_filters(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.5]})
        assert mmp_deltas(frame, "smiles", "y", min_pairs=2).empty

    def test_empty_frame_keeps_schema(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1"], "y": [1.0]})
        deltas = mmp_deltas(frame, "smiles", "y")
        assert list(deltas.columns) == [
            "context",
            "left_sub",
            "right_sub",
            "n_pairs",
            "mean_delta",
            "std_delta",
            "mean_activity",
        ]

    def test_std_is_zero_for_singletons(self):
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N"], "y": [5.0, 6.5]})
        deltas = mmp_deltas(frame, "smiles", "y")
        assert float(deltas["std_delta"].iloc[0]) == 0.0


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #
class TestResultTypes:
    def test_matched_pair_as_dict(self):
        pair = MatchedPair("CC", "O", "N", "CCO", "CCN", 5.0, 6.0, 1.0)
        row = pair.as_dict()
        assert row["context"] == "CC"
        assert row["delta"] == 1.0
        assert set(row) == set(MATCHED_PAIR_COLUMNS)

    def test_rgroup_as_dict(self):
        group = RGroup(smiles="c1ccccc1O", groups={1: "O"}, valid=True)
        row = group.as_dict()
        assert row == {"smiles": "c1ccccc1O", "valid": True, "R1": "O"}


# --------------------------------------------------------------------------- #
# R-group decomposition (needs RDKit)
# --------------------------------------------------------------------------- #
class TestRGroupDecomposition:
    def test_single_attachment(self):
        pytest.importorskip("rdkit")
        frame = pd.DataFrame(
            {"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1"], "y": [5.0, 6.0, 4.0]}
        )
        table = rgroup_table(frame, "smiles", "c1ccccc1[*:1]", activity_col="y")
        assert table["valid"].tolist() == [True, True, True]
        assert table["R1"].tolist() == ["O", "N", "[H]"]
        assert table["y"].tolist() == [5.0, 6.0, 4.0]

    def test_unmatched_molecule_is_invalid(self):
        pytest.importorskip("rdkit")
        frame = pd.DataFrame({"smiles": ["c1ccccc1O", "CCCC"]})
        table = rgroup_table(frame, "smiles", "c1ccccc1[*:1]")
        assert table["valid"].tolist() == [True, False]

    def test_missing_smiles_column_raises(self):
        pytest.importorskip("rdkit")
        frame = pd.DataFrame({"other": ["c1ccccc1O"]})
        with pytest.raises(DataValidationError, match="not in frame"):
            rgroup_table(frame, "smiles", "c1ccccc1[*:1]")

    def test_missing_activity_column_raises(self):
        pytest.importorskip("rdkit")
        frame = pd.DataFrame({"smiles": ["c1ccccc1O"]})
        with pytest.raises(DataValidationError, match="not in frame"):
            rgroup_table(frame, "smiles", "c1ccccc1[*:1]", activity_col="y")

    def test_bad_core_raises(self):
        pytest.importorskip("rdkit")
        frame = pd.DataFrame({"smiles": ["c1ccccc1O"]})
        with pytest.raises(DataValidationError, match="neither valid"):
            rgroup_table(frame, "smiles", "this is not a core ][")

    def test_two_attachments(self):
        pytest.importorskip("rdkit")
        # meta-disubstituted benzene: two marked positions
        core = "[*:1]c1cccc([*:2])c1"
        frame = pd.DataFrame({"smiles": ["Oc1cccc(N)c1"]})
        table = rgroup_table(frame, "smiles", core)
        assert table["valid"].tolist() == [True]
        assert {"R1", "R2"}.issubset(table.columns)

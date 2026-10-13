"""Unit tests for similarity math and validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.chem.similarity import (
    tanimoto_from_bitarrays,
    tanimoto_from_sets,
    tanimoto_matrix,
)
from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.utils.validation import (
    require_columns,
    validate_concentration_series,
    validate_positive,
    validate_smiles_syntax,
)


class TestTanimotoSets:
    def test_identical_sets(self) -> None:
        assert tanimoto_from_sets({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert tanimoto_from_sets({"a", "b"}, {"c", "d"}) == 0.0

    def test_known_overlap(self) -> None:
        # |{b,c}| / |{a,b,c,d}| = 2/4
        assert tanimoto_from_sets({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert tanimoto_from_sets(set(), set()) == 1.0


class TestTanimotoBitarrays:
    def test_matches_set_math(self) -> None:
        a = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        b = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        assert tanimoto_from_bitarrays(a, b) == pytest.approx(2 / 4)

    def test_identical_bits(self) -> None:
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert tanimoto_from_bitarrays(bits, bits) == 1.0

    def test_both_zero(self) -> None:
        zeros = np.zeros(8, dtype=np.uint8)
        assert tanimoto_from_bitarrays(zeros, zeros) == 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(DataValidationError):
            tanimoto_from_bitarrays(np.zeros(4), np.zeros(5))


class TestTanimotoMatrix:
    def test_agrees_with_pairwise_function(self) -> None:
        rng = np.random.default_rng(0)
        reference = (rng.random((4, 32)) > 0.7).astype(np.uint8)
        targets = (rng.random((5, 32)) > 0.7).astype(np.uint8)
        matrix = tanimoto_matrix(reference, targets)
        assert matrix.shape == (4, 5)
        for i in range(4):
            for j in range(5):
                assert matrix[i, j] == pytest.approx(
                    tanimoto_from_bitarrays(reference[i], targets[j])
                )

    def test_range_bounds(self) -> None:
        rng = np.random.default_rng(1)
        matrix = tanimoto_matrix(
            (rng.random((10, 64)) > 0.5).astype(np.uint8),
            (rng.random((10, 64)) > 0.5).astype(np.uint8),
        )
        assert matrix.min() >= 0.0 and matrix.max() <= 1.0

    def test_dimension_validation(self) -> None:
        with pytest.raises(DataValidationError):
            tanimoto_matrix(np.zeros((2, 4)), np.zeros((3, 5)))


class TestRDKitGatedFeatures:
    """RDKit-dependent helpers raise actionable errors when missing."""

    def test_similarity_scan_needs_rdkit(self) -> None:
        try:
            import rdkit  # noqa: F401

            pytest.skip("RDKit installed; gated behaviour not exercised")
        except ImportError:
            pass
        from pharmaml.chem.similarity import similarity_scan

        with pytest.raises(DependencyError, match="pip install"):
            similarity_scan(["CCO"], ["CCC"], threshold=0.5)


class TestValidationHelpers:
    def test_validate_positive(self) -> None:
        assert validate_positive(3.5, "x") == 3.5
        assert validate_positive(0, "x", allow_zero=True) == 0.0
        for bad in (-1, float("nan"), float("inf"), "text"):
            with pytest.raises(DataValidationError):
                validate_positive(bad, "x")

    def test_validate_smiles_syntax(self) -> None:
        assert validate_smiles_syntax("CCO")
        assert validate_smiles_syntax("c1ccccc1")
        assert not validate_smiles_syntax("CC(O")     # unbalanced
        assert not validate_smiles_syntax("")          # empty
        assert not validate_smiles_syntax("CC$%^")     # illegal chars

    def test_require_columns(self) -> None:
        frame = pd.DataFrame({"a": [1], "b": [2]})
        require_columns(frame, ["a", "b"])
        with pytest.raises(DataValidationError, match="Missing required"):
            require_columns(frame, ["a", "c", "d"])

    def test_validate_concentration_series(self) -> None:
        t, c = validate_concentration_series([0, 1, 2], [1.0, 2.0, 1.5])
        assert t.tolist() == [0, 1, 2]
        with pytest.raises(DataValidationError):
            validate_concentration_series([0, 1, 1], [1.0, 1.0, 1.0])
        with pytest.raises(DataValidationError):
            validate_concentration_series([0, 1], [1.0])  # length mismatch
        with pytest.raises(DataValidationError):
            validate_concentration_series([0], [1.0])     # too few points

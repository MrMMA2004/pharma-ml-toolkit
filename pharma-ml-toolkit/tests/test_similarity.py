"""Unit tests for similarity math and validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.chem.similarity import (
    similarity_scan,
    tanimoto_from_bitarrays,
    tanimoto_from_sets,
    tanimoto_matrix,
    top_matches,
)
from pharmaml.exceptions import DataValidationError, DependencyError

from pharmaml.utils.validation import (
    require_columns,
    validate_concentration_series,
    validate_positive,
    validate_smiles_syntax,
)

try:  # RDKit gates only the scan-level regression tests at the bottom
    import rdkit  # noqa: F401

    HAS_RDKIT = True
except ImportError:  # pragma: no cover - exercised on chem-extra-less CI
    HAS_RDKIT = False


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


# ---------------------------------------------------------------------------
# Round 100x-19-a regression tests (fail-first against pristine sources)
# ---------------------------------------------------------------------------


class TestTanimotoNonFiniteInput:
    """NaN in a float "bit array" silently counted as a SET bit."""

    def test_nan_in_float_bitarray_raises(self) -> None:
        a = np.array([1.0, np.nan, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(DataValidationError, match="non-finite"):
            tanimoto_from_bitarrays(a, b)

    def test_inf_in_float_bitarray_raises(self) -> None:
        with pytest.raises(DataValidationError, match="non-finite"):
            tanimoto_from_bitarrays(np.array([1.0, np.inf]), np.array([1.0, 0.0]))

    def test_nan_in_matrix_raises(self) -> None:
        with pytest.raises(DataValidationError, match="non-finite"):
            tanimoto_matrix(np.array([[1.0, np.nan]]), np.array([[1.0, 0.0]]))
        with pytest.raises(DataValidationError, match="non-finite"):
            tanimoto_matrix(np.array([[1.0, 0.0]]), np.array([[np.nan, 0.0]]))

    def test_finite_float_arrays_still_accepted(self) -> None:
        # Disclosed pin: float 0/1 arrays remain valid input.
        assert tanimoto_from_bitarrays(
            np.array([1.0, 0.0]), np.array([1.0, 1.0])
        ) == pytest.approx(0.5)

    def test_matrix_zero_row_convention_matches_pairwise(self) -> None:
        # Disclosed pin: both-empty -> 1.0, one-empty -> 0.0, everywhere.
        ref = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        tgt = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.uint8)
        matrix = tanimoto_matrix(ref, tgt)
        assert matrix[0, 0] == pytest.approx(0.0)
        assert matrix[1, 0] == pytest.approx(1.0)
        for i in range(2):
            for j in range(2):
                assert matrix[i, j] == pytest.approx(
                    tanimoto_from_bitarrays(ref[i], tgt[j])
                )


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for scan-level tests")
class TestSimilarityScanRegression:
    """Positional (not label-based) SMILES lookup + sequence contract."""

    def test_series_reference_uses_positional_order(self) -> None:
        # Series index [1, 0, 2]: positional row 0 is 'CCO' (label 1).
        refs = pd.Series(["CCO", "CCN", "CCF"], index=[1, 0, 2])
        result = similarity_scan(refs, ["CCO"], threshold=0.99, fingerprint="morgan")
        assert result["reference"].tolist() == ["CCO"]
        assert result["reference_index"].tolist() == [0]
        assert result["similarity"].iloc[0] == pytest.approx(1.0)

    def test_series_reference_without_label_zero_still_works(self) -> None:
        refs = pd.Series(["CCO", "CCN"], index=[10, 20])
        result = similarity_scan(refs, ["CCO"], threshold=0.99, fingerprint="morgan")
        assert result["reference"].tolist() == ["CCO"]

    def test_series_target_uses_positional_order(self) -> None:
        targets = pd.Series(["CCO", "CCN"], index=[1, 0])
        result = similarity_scan(["CCO"], targets, threshold=0.99, fingerprint="morgan")
        assert result["target"].tolist() == ["CCO"]
        assert result["target_index"].tolist() == [0]

    def test_generator_reference_supported(self) -> None:
        result = similarity_scan(
            iter(["CCO"]), ["CCO", "CCN"], threshold=0.5, fingerprint="morgan"
        )
        assert result["reference"].tolist() == ["CCO"]
        assert result["target"].tolist() == ["CCO"]

    def test_bare_string_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            similarity_scan("CCO", ["CCO"])
        with pytest.raises(DataValidationError, match="single string"):
            similarity_scan(["CCO"], "CCO")

    def test_long_format_and_descending_order(self) -> None:
        # Disclosed pin: tidy long format, sorted by descending similarity.
        result = similarity_scan(
            ["CCO", "CCN"], ["CCO", "CCN", "CCF"], threshold=0.0, fingerprint="morgan"
        )
        assert list(result.columns) == [
            "reference",
            "target",
            "reference_index",
            "target_index",
            "similarity",
        ]
        sims = result["similarity"].tolist()
        assert sims == sorted(sims, reverse=True)
        identity = result[(result["reference"] == "CCO") & (result["target"] == "CCO")]
        assert identity["similarity"].iloc[0] == pytest.approx(1.0)

    def test_equal_similarity_ties_keep_library_order(self) -> None:
        # Disclosed pin (stable sort): ties preserve library order, so the
        # k-boundary of top_matches is predictable.
        library = ["CCO", "CCO", "CCN"]
        result = top_matches("CCO", library, k=3, fingerprint="morgan")
        assert result["target"].tolist() == ["CCO", "CCO", "CCN"]

    def test_threshold_bounds_validation(self) -> None:
        with pytest.raises(DataValidationError, match="threshold"):
            similarity_scan(["CCO"], ["CCO"], threshold=1.5)


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit required for top_matches")
class TestTopMatchesRegression:
    """Invalid queries must raise, not return a misleading empty frame."""

    def test_invalid_query_raises(self) -> None:
        with pytest.raises(DataValidationError, match="C1CC"):
            top_matches("C1CC(` bad", ["CCO", "CCN"])

    def test_empty_query_raises(self) -> None:
        # Pre-fix this was even worse: an all-zero fingerprint whose
        # similarity to everything is 0.0, so rows were RETURNED.
        with pytest.raises(DataValidationError, match="query SMILES"):
            top_matches("", ["CCO"])

    def test_bare_string_library_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            top_matches("CCO", "CCN")

    def test_top_matches_returns_k_sorted(self) -> None:
        # Disclosed pin: best match first, at most k rows.
        result = top_matches("CCO", ["CCF", "CCN", "CCO"], k=2, fingerprint="morgan")
        assert result["target"].tolist()[0] == "CCO"
        assert len(result) == 2
        assert result["similarity"].is_monotonic_decreasing

    def test_radius_parameter_is_forwarded(self) -> None:
        library = ["c1ccccc1O", "CCCCCC", "CCO"]
        via_scan = (
            similarity_scan(
                ["c1ccccc1"],
                library,
                threshold=0.0,
                fingerprint="morgan",
                radius=1,
                n_bits=512,
            )
            .head(3)
            .reset_index(drop=True)
        )
        via_top = top_matches(
            "c1ccccc1", library, k=3, fingerprint="morgan", radius=1, n_bits=512
        )
        pd.testing.assert_frame_equal(via_top, via_scan)

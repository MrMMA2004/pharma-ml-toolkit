"""Tests for :mod:`pharmaml.chem.descriptors` (RDKit-backed paths).

Round 100x-19-a regression suite.  The first classes collect fail-first
tests for defects found in the adversarial review (run against the
pristine sources before any fix); the pinning classes at the bottom
guard behaviour that was already correct but non-obvious.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from pharmaml.chem.descriptors import (
    compute_descriptors,
    compute_fingerprints,
    compute_maccs,
)
from pharmaml.exceptions import DataValidationError

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
INVALID = "not_a_smiles"


class TestDescriptorSetValidation:
    """Fail-first: unknown ``descriptor_set`` must not silently fall back."""

    def test_unknown_descriptor_set_raises(self) -> None:
        with pytest.raises(DataValidationError, match="descriptor_set") as excinfo:
            compute_descriptors([ETHANOL], descriptor_set="mordeed", progress=False)
        assert "mordeed" in str(excinfo.value)

    def test_unknown_descriptor_set_error_lists_valid_choices(self) -> None:
        with pytest.raises(DataValidationError, match="rdkit_all.*morgan.*mordred"):
            compute_descriptors([ETHANOL], descriptor_set="typo", progress=False)


class TestMorganDescriptorSet:
    """Fail-first: ``descriptor_set='morgan'`` crashed on every input."""

    def test_morgan_descriptor_set_roundtrip(self) -> None:
        frame = compute_descriptors([ETHANOL], descriptor_set="morgan", progress=False)
        assert frame.shape == (1, 2049)
        assert frame["SMILES"].tolist() == [ETHANOL]
        fp_columns = [c for c in frame.columns if c != "SMILES"]
        assert fp_columns[0] == "FP000"
        assert fp_columns[-1] == "FP2047"
        values = frame[fp_columns].iloc[0].to_numpy()
        assert values.sum() > 0  # ethanol has radius-2 environments

    def test_morgan_descriptor_set_matches_rdkit_counts(self) -> None:
        # The frame must hold the *count* fingerprint of GetHashedMorganFingerprint
        # (radius 2, 2048 bits) exactly -- value-for-value at each index.
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        smiles = "CCCC"
        frame = compute_descriptors([smiles], descriptor_set="morgan", progress=False)
        row = frame[[c for c in frame.columns if c != "SMILES"]].iloc[0].to_numpy()
        nonzero = rdMolDescriptors.GetHashedMorganFingerprint(
            Chem.MolFromSmiles(smiles), radius=2, nBits=2048
        ).GetNonzeroElements()
        for idx, count in nonzero.items():
            assert row[idx] == count
        assert row.sum() == sum(nonzero.values())
        # A count fingerprint (not binary): repeated environments occur.
        assert (row > 1).any()


class TestBlankSmilesAreInvalid:
    """Fail-first: RDKit parses ``''`` as a *valid* 0-atom molecule.

    Blank cells must be reported as invalid like any other parse failure
    (consistent with ``utils.validation.validate_smiles_syntax``), not
    crash downstream with an opaque ``ValueError`` or silently produce
    all-zero descriptor/fingerprint rows.
    """

    def test_empty_string_invalid_in_descriptors(self) -> None:
        frame = compute_descriptors(["", ETHANOL], progress=False)
        assert frame.index.tolist() == [1]
        assert frame["SMILES"].tolist() == [ETHANOL]

    def test_empty_string_invalid_in_fingerprints(self) -> None:
        matrix, valid = compute_fingerprints(["", ETHANOL])
        assert valid == [1]
        assert matrix.shape == (1, 2048)

    def test_empty_string_invalid_in_maccs(self) -> None:
        matrix, valid = compute_maccs([ETHANOL, ""])
        assert valid == [0]
        assert matrix.shape == (1, 167)


class TestSequenceInputContract:
    """Fail-first: generators crash, bare strings silently explode to chars."""

    def test_generator_input_supported(self) -> None:
        frame = compute_descriptors(iter([ETHANOL, BENZENE]), progress=False)
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE]
        assert frame.index.tolist() == [0, 1]

    def test_bare_string_rejected_descriptors(self) -> None:
        with pytest.raises(DataValidationError, match="single string") as excinfo:
            compute_descriptors(ETHANOL, progress=False)
        assert ETHANOL in str(excinfo.value)

    def test_bare_string_rejected_fingerprints(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            compute_fingerprints(ETHANOL)

    def test_bare_string_rejected_maccs(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            compute_maccs(ETHANOL)

    def test_long_bare_string_preview_is_truncated(self) -> None:
        long_smiles = "C" * 200
        with pytest.raises(DataValidationError, match="single string") as excinfo:
            compute_fingerprints(long_smiles)
        assert len(str(excinfo.value)) < 200  # no 200-char dump in the error


class TestInvalidSmilesAccounting:
    """Fail-first: every public entry point must name dropped indices."""

    @pytest.fixture(autouse=True)
    def _visible_logs(self):
        """Make this module's warnings visible to ``caplog``.

        Two independent silencers must be lifted (established suite
        conventions, cf. ``tests/test_docking.py`` and
        ``tests/test_split_categorical.py``): sibling modules call
        ``logging.disable(CRITICAL)`` at import time, and CLI tests call
        ``configure_logging`` which sets ``propagate=False`` on the
        ``pharmaml`` root logger while pytest's caplog handler lives on
        the true root logger.
        """
        prev_disable = logging.root.manager.disable
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        logging.disable(logging.NOTSET)
        pharma_root.propagate = True
        yield
        pharma_root.propagate = prev_propagate
        logging.disable(prev_disable)

    def test_maccs_warns_about_invalid_smiles(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            _matrix, valid = compute_maccs([ETHANOL, INVALID])
        assert valid == [0]
        assert "index 1" in caplog.text

    def test_fingerprints_warning_names_offending_index(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            compute_fingerprints([ETHANOL, INVALID, BENZENE])
        assert "index 1" in caplog.text

    def test_descriptors_warning_names_offending_index(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            compute_descriptors([ETHANOL, INVALID], progress=False)
        assert "index 1" in caplog.text


# ---------------------------------------------------------------------------
# Pinning tests: behaviour that was already correct (disclosed, non-obvious)
# ---------------------------------------------------------------------------


class TestComputeDescriptorsPins:
    def test_rdkit_all_shape_and_molwt(self) -> None:
        frame = compute_descriptors([ETHANOL], progress=False)
        assert frame.shape[0] == 1
        assert "MolWt" in frame.columns
        assert frame["MolWt"].iloc[0] == pytest.approx(46.069, abs=1e-2)
        assert frame.index.name == "original_index"

    def test_series_with_shuffled_index_is_positional(self) -> None:
        # A DataFrame column with a non-default index: positions, not labels.
        series = pd.Series([ETHANOL, BENZENE], index=[5, 0])
        frame = compute_descriptors(series, progress=False)
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE]
        assert frame.index.tolist() == [0, 1]

    def test_invalid_smiles_dropped_from_output(self) -> None:
        frame = compute_descriptors([INVALID, ETHANOL, BENZENE], progress=False)
        assert frame.index.tolist() == [1, 2]
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE]


class TestComputeFingerprintsPins:
    def test_morgan_bits_shape_and_dtype(self) -> None:
        matrix, valid = compute_fingerprints([ETHANOL, BENZENE], kind="morgan", n_bits=512)
        assert matrix.shape == (2, 512)
        assert matrix.dtype == np.uint8
        assert valid == [0, 1]
        assert matrix[0].sum() > 0

    def test_morgan_counts_dtype_and_values(self) -> None:
        matrix, _ = compute_fingerprints(["CCCC"], kind="morgan", as_counts=True, n_bits=2048)
        assert matrix.dtype == np.uint32
        assert matrix.max() > 1  # count fingerprint, not binary

    def test_rdkit_and_atom_pair_kinds(self) -> None:
        for kind in ("rdkit", "atom_pair"):
            matrix, valid = compute_fingerprints([ETHANOL], kind=kind, n_bits=256)
            assert matrix.shape == (1, 256)
            assert valid == [0]

    def test_maccs_shape(self) -> None:
        matrix, valid = compute_maccs([ETHANOL, BENZENE])
        assert matrix.shape == (2, 167)
        assert matrix.dtype == np.uint8
        assert valid == [0, 1]

    def test_parameter_validation(self) -> None:
        with pytest.raises(DataValidationError, match="radius"):
            compute_fingerprints([ETHANOL], radius=-1)
        with pytest.raises(DataValidationError, match="n_bits"):
            compute_fingerprints([ETHANOL], n_bits=0)
        with pytest.raises(DataValidationError, match="kind"):
            compute_fingerprints([ETHANOL], kind="dragon")

    def test_all_invalid_smiles_yields_empty_matrix(self) -> None:
        matrix, valid = compute_fingerprints([INVALID, "also bad"])
        assert matrix.shape == (0, 2048)
        assert valid == []

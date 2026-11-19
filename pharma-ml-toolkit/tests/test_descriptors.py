"""Tests for :mod:`pharmaml.chem.descriptors` (RDKit-backed paths).

Round 100x-19-a regression suite.  The first classes collect fail-first
tests for defects found in the adversarial review (run against the
pristine sources before any fix); the pinning classes at the bottom
guard behaviour that was already correct but non-obvious.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from pharmaml.chem.descriptors import (
    D3D_COLUMNS,
    compute_3d_descriptors,
    compute_descriptors,
    compute_fingerprints,
    compute_maccs,
)
from pharmaml.exceptions import DataValidationError

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "not_a_smiles"

MORDED_INSTALLED = importlib.util.find_spec("mordred") is not None


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


class TestZeroHeavyAtomMolecules:
    """Fail-first (100x-41-G): a single ``[H]`` crashed the whole batch.

    RDKit's SPS descriptor divides by ``GetNumHeavyAtoms()`` -- zero for
    ``[H]``/``[H][H]`` -- so ``compute_descriptors`` leaked a raw
    ``ZeroDivisionError`` and lost every *other* row of the batch as
    collateral (fail-first evidence: /tmp/task41g/probe_hydrogen_FAIL.txt).
    """

    @pytest.fixture(autouse=True)
    def _visible_logs(self):
        """Same two silencers as TestInvalidSmilesAccounting (suite
        convention, cf. tests/test_format_converters.py): module-level
        ``logging.disable`` plus ``propagate=False`` on the pharmaml root
        logger both hide module warnings from caplog until lifted."""
        prev_disable = logging.root.manager.disable
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        logging.disable(logging.NOTSET)
        pharma_root.propagate = True
        yield
        pharma_root.propagate = prev_propagate
        logging.disable(prev_disable)

    def test_hydrogen_dropped_from_mixed_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            frame = compute_descriptors([ETHANOL, "[H]", BENZENE], progress=False)
        # the [H] row is dropped like an invalid SMILES, index stays aligned
        assert frame.index.tolist() == [0, 2]
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE]
        assert frame.index.name == "original_index"
        assert "index 1" in caplog.text
        assert "[H]" in caplog.text
        assert "descriptor computation failed" in caplog.text

    def test_hydrogen_drop_composes_with_invalid_smiles(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # parse-drop (index 1) and descriptor-drop (index 0) in one batch
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            frame = compute_descriptors(["[H]", INVALID, ETHANOL], progress=False)
        assert frame.index.tolist() == [2]
        assert frame["SMILES"].tolist() == [ETHANOL]
        assert "index 0" in caplog.text
        assert "index 1" in caplog.text

    @pytest.mark.parametrize("batch", [["[H]"], ["[H]", "[H][H]"]])
    def test_all_zero_heavy_atoms_raise_typed_error(self, batch: list[str]) -> None:
        # every row dropped -> the module's typed empty-result contract,
        # mirroring the all-invalid-SMILES DataValidationError
        with pytest.raises(DataValidationError, match="every valid molecule"):
            compute_descriptors(batch, progress=False)

    def test_fingerprints_and_maccs_on_hydrogen_unchanged(self) -> None:
        # the drop is rdkit_all-only: fingerprint paths compute on [H]
        matrix, valid = compute_fingerprints(["[H]"], n_bits=64)
        assert matrix.shape == (1, 64)
        assert valid == [0]
        maccs, maccs_valid = compute_maccs(["[H]"])
        assert maccs.shape == (1, 167)
        assert maccs_valid == [0]


class TestFingerprintParameterTypes:
    """Fail-first (100x-41-G): non-int ``radius``/``n_bits`` leaked raw errors.

    str/None raised a raw ``TypeError`` from the guard comparison itself,
    floats passed the range guard and blew up in the Boost.Python binding
    (``ArgumentError`` plus a C++ signature dump), and bools were silently
    read as ``1``/``0`` (radius) or raw-errored deeper (n_bits).  All now
    raise the documented ``DataValidationError``; bools are rejected by
    decision -- ``True``/``False`` are nonsense as a radius/bit count
    (fail-first evidence: /tmp/task41g/probe_params_FAIL.txt).
    """

    @pytest.mark.parametrize("value", ["2", None, 2.5, True, False])
    def test_non_int_radius_raises_typed(self, value: object) -> None:
        with pytest.raises(
            DataValidationError, match="radius must be an integer"
        ) as excinfo:
            compute_fingerprints([ETHANOL], radius=value)
        assert repr(value) in str(excinfo.value)

    @pytest.mark.parametrize("value", ["64", None, 64.0, 16.5, True, False])
    def test_non_int_n_bits_raises_typed(self, value: object) -> None:
        with pytest.raises(
            DataValidationError, match="n_bits must be an integer"
        ) as excinfo:
            compute_fingerprints([ETHANOL], n_bits=value)
        assert repr(value) in str(excinfo.value)

    def test_numpy_integers_accepted(self) -> None:
        # numpy ints pass the guard and are normalised to plain int (the
        # Boost bindings reject numpy scalar types -- raw ArgumentError
        # pre-fix)
        matrix, valid = compute_fingerprints(
            [ETHANOL], radius=np.int64(2), n_bits=np.int64(64)
        )
        assert matrix.shape == (1, 64)
        assert valid == [0]

    def test_range_guards_unchanged_for_plain_ints(self) -> None:
        # boundary values behave exactly as before the int-ness guards
        assert compute_fingerprints([ETHANOL], radius=0, n_bits=1)[0].shape == (1, 1)
        with pytest.raises(DataValidationError, match="radius must be >= 0"):
            compute_fingerprints([ETHANOL], radius=-1)
        with pytest.raises(DataValidationError, match="n_bits out of range"):
            compute_fingerprints([ETHANOL], n_bits=0)


class TestEarlyValidationOrdering:
    """Fail-first (100x-41-G): empty input / unknown ``descriptor_set`` were
    validated only after the parse loop and the tqdm spin-up (finding G5).

    Error *types* are unchanged (DataValidationError); only ordering and
    message quality improve (fail-first evidence:
    /tmp/task41g/probe_ordering_FAIL.txt).
    """

    def test_empty_list_message_names_the_problem(self) -> None:
        # an empty input is a caller bug, not a parse outcome -- the old
        # message ("No valid SMILES found in the input") misled about both
        with pytest.raises(DataValidationError, match="empty") as excinfo:
            compute_descriptors([], progress=False)
        assert "No valid SMILES" not in str(excinfo.value)

    def test_empty_generator_treated_like_empty_list(self) -> None:
        with pytest.raises(DataValidationError, match="empty"):
            compute_descriptors(iter([]), progress=False)

    def test_descriptor_set_validated_before_parsing(self) -> None:
        # a batch whose SMILES are ALL invalid must still surface the
        # descriptor_set error (pre-fix the parse accounting won the race)
        with pytest.raises(DataValidationError, match="descriptor_set"):
            compute_descriptors(
                [INVALID, "also bad"], descriptor_set="typo", progress=False
            )

    def test_no_progress_bar_spins_up_before_validation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(DataValidationError, match="descriptor_set"):
            compute_descriptors([ETHANOL] * 20, descriptor_set="typo", progress=True)
        assert "Descriptors" not in capsys.readouterr().err

    def test_all_invalid_non_empty_batch_error_unchanged(self) -> None:
        # the neighbouring contract is untouched: a NON-empty all-invalid
        # batch still raises the parse-accounting error
        with pytest.raises(DataValidationError, match="No valid SMILES found"):
            compute_descriptors([INVALID, "also bad"], progress=False)


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

    def test_avalon_bits_shape_and_pinned_sum(self) -> None:
        # The rar's Descriptor classification/regression.py feature
        # unions folded 512-bit Avalon fingerprints -- the binding now
        # hashes straight to n_bits.
        matrix, valid = compute_fingerprints(
            [ETHANOL, BENZENE], kind="avalon", n_bits=512
        )
        assert matrix.shape == (2, 512)
        assert matrix.dtype == np.uint8
        assert valid == [0, 1]
        assert matrix.sum(axis=1).tolist() == [8, 8]  # pinned Avalon bits
        # same popcount, different bits: ethanol vs benzene set none of
        # the same positions.
        ethanol_idx = set(np.nonzero(matrix[0])[0].tolist())
        benzene_idx = set(np.nonzero(matrix[1])[0].tolist())
        assert ethanol_idx.isdisjoint(benzene_idx)

    def test_avalon_counts_dtype_and_values(self) -> None:
        # as_counts for Avalon upgrades the rar's get_avalon_count_fp
        # hand-folded 512-wide dense vector.
        matrix, valid = compute_fingerprints(
            [ETHANOL], kind="avalon", n_bits=512, as_counts=True
        )
        assert matrix.shape == (1, 512)
        assert matrix.dtype == np.uint32
        assert valid == [0]
        assert int(matrix.sum()) == 7  # ethanol: 7 nonzero count entries

    def test_avalon_bits_differ_between_molecules(self) -> None:
        ethanol, benzene = compute_fingerprints(
            [ETHANOL, BENZENE], kind="avalon", n_bits=512
        )
        assert not np.array_equal(ethanol, benzene)

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

    def test_as_counts_rejected_for_non_morgan_kinds(self) -> None:
        # Regression (100x-22-c): as_counts=True used to be silently
        # ignored for non-Morgan kinds, returning binary uint8 bits
        # where the caller asked for counts (fail-first evidence in
        # /tmp/task22c/failfirst_ascounts.txt).
        for kind in ("rdkit", "atom_pair"):
            with pytest.raises(DataValidationError, match="as_counts=True is only supported"):
                compute_fingerprints([ETHANOL], kind=kind, as_counts=True, n_bits=256)

    def test_all_invalid_smiles_yields_empty_matrix(self) -> None:
        matrix, valid = compute_fingerprints([INVALID, "also bad"])
        assert matrix.shape == (0, 2048)
        assert valid == []


@pytest.mark.skipif(not MORDED_INSTALLED, reason="mordred is optional (requirements.txt)")
class TestMordredDescriptorSet:
    """Fail-first (100x-52): the mordred frame was unusable.

    Two defects, both live on this full-deps venv: (1) the frame's
    column names were mordred *Descriptor instances*, not strings --
    ``frame["MW"]`` raised ``KeyError`` and even iterating
    ``frame.columns`` exploded inside pandas' ``apply_if_callable``
    (Descriptor instances are callable, so pandas called them *on the
    frame*); (2) failed cells were raw mordred ``Error``/``Missing``
    objects rather than NaN, leaving object-dtype columns that no
    downstream ML path can consume.
    """

    @pytest.fixture(autouse=True)
    def _visible_logs(self):
        """Lift the two log silencers so ``caplog`` sees the module."""
        prev_disable = logging.root.manager.disable
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        logging.disable(logging.NOTSET)
        pharma_root.propagate = True
        yield
        pharma_root.propagate = prev_propagate
        logging.disable(prev_disable)

    def test_column_names_are_strings(self) -> None:
        frame = compute_descriptors([BENZENE], descriptor_set="mordred", progress=False)
        assert frame.shape[0] == 1
        assert frame.shape[1] > 1500  # the full 2D suite, not a subset
        assert all(isinstance(col, str) for col in frame.columns)

    def test_columns_indexable_by_name_with_pinned_values(self) -> None:
        # The money probe: name-based access must simply work.  Values
        # pinned against a direct mordred census (RDKit 2026.09.1 /
        # mordred 1.2.0): benzene MW ~78.047, nH 6, nRing 1.
        frame = compute_descriptors([BENZENE], descriptor_set="mordred", progress=False)
        assert frame["MW"].iloc[0] == pytest.approx(78.046950192)
        assert int(frame["nH"].iloc[0]) == 6
        assert int(frame["nRing"].iloc[0]) == 1

    def test_failed_cells_become_nan_not_error_objects(self) -> None:
        # A sodium ion makes mordred return hundreds of Missing/Error
        # cells; every one must arrive as float NaN, and no non-SMILES
        # column may be object dtype (the old frame was full of them).
        frame = compute_descriptors(["[Na+]"], descriptor_set="mordred", progress=False)
        numeric = frame.drop(columns=["SMILES"])
        assert not any(numeric.dtypes == object)
        nan_cells = int(numeric.isna().to_numpy().sum())
        assert nan_cells > 100  # the failure verdicts really are NaN now
        assert np.isfinite(numeric["MW"].iloc[0])  # MW itself survives

    def test_mixed_batch_frame_is_ml_ready(self) -> None:
        frame = compute_descriptors(
            [ETHANOL, BENZENE, "[Na+].[Cl-]"], descriptor_set="mordred", progress=False
        )
        assert frame.shape[0] == 3
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE, "[Na+].[Cl-]"]
        assert frame.index.tolist() == [0, 1, 2]
        numeric = frame.drop(columns=["SMILES"])
        coerced = numeric.astype(float)  # the ML-readiness contract
        means = coerced.mean(axis=0)
        assert np.isfinite(means["MW"])

    def test_aggregate_warning_names_cell_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            compute_descriptors([ETHANOL, "[Na+]"], descriptor_set="mordred", progress=False)
        text = caplog.text
        assert "descriptor cells" in text
        assert "Error" in text and "Missing" in text
        # The aggregate count must reflect BOTH molecules (2 x 1613 cells
        # with hundreds of failures each), not just the first.
        import re

        counts = re.findall(r"(\d+) of (\d+) descriptor cells", text)
        assert counts, text
        failed, total = (int(value) for value in counts[0])
        assert failed > 600
        assert total == 2 * 1613

    def test_warning_fires_once_per_call_not_per_molecule(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The aggregate contract: ONE warning for the whole batch, no
        # matter how many molecules carry failure verdicts (a per-
        # molecule warning would spam 10k lines on a real library).
        with caplog.at_level(logging.WARNING, logger="pharmaml.chem.descriptors"):
            compute_descriptors(
                [ETHANOL, BENZENE, "[Na+]"], descriptor_set="mordred", progress=False
            )
        records = [r for r in caplog.records if "descriptor cells" in r.getMessage()]
        assert len(records) == 1, caplog.text


# ---------------------------------------------------------------------------
# 3D shape descriptors (xxx.ipynb cell 6's RDKit half, 100x-147)
# ---------------------------------------------------------------------------


class TestCompute3dDescriptors:
    """The ETKDG/MMFF/UFF shape-descriptor pipeline and its contracts.

    Every embedding runs at the fixed ETKDG seed 42, so all value pins
    below are exact on this RDKit build (2026.09.1) and deterministic
    across runs.
    """

    def test_d3d_columns_contract(self) -> None:
        assert D3D_COLUMNS == (
            "Asphericity",
            "Eccentricity",
            "PMI1",
            "PMI2",
            "PMI3",
            "RadiusOfGyration",
            "SpherocityIndex",
            "NPR1",
            "NPR2",
        )

    def test_aspirin_shape_and_pinned_values(self) -> None:
        frame = compute_3d_descriptors([ASPIRIN], progress=False)
        assert list(frame.columns) == ["SMILES", *D3D_COLUMNS]
        assert frame.index.name == "original_index"
        row = frame[list(D3D_COLUMNS)].iloc[0]
        # Pinned against the ETKDG(42)/MMFF94s pipeline on this build.
        # (r249 environment: PMI1 463.515414 / RoG 2.428345 / SI 0.09743;
        #  r263 environment (rdkit 2026.03.4) shifts PMI1's 6th digit —
        #  the pins below carry BOTH tolerance heads: same embedding to
        #  ~3e-4 relative, which is what the ETKDG(42) seed guarantees
        #  across rdkit point releases. r296 third head: the Windows
        #  runner's MSVC-built rdkit shifts PMI1 to 463.51584 (5.4e-4
        #  over the pin) — same-class FP divergence as the rdkit point
        #  releases, so PMI1's tolerance widens to 5e-3 (RoG/SI stay:
        #  their relative shift stays an order below their pins).)
        assert float(row["PMI1"]) == pytest.approx(463.5153, abs=5e-3)
        assert float(row["RadiusOfGyration"]) == pytest.approx(2.428345, abs=1e-5)
        assert float(row["SpherocityIndex"]) == pytest.approx(0.09743, abs=1e-4)

    def test_aspirin_descriptor_sanity(self) -> None:
        values = compute_3d_descriptors([ASPIRIN], progress=False)[
            list(D3D_COLUMNS)
        ].iloc[0].to_numpy()
        assert np.isfinite(values).all()  # 9 finite ...
        assert (values >= 0.0).all()  # ... positive-ish shape descriptors
        # Principal moments are ordered; both NPRs live in (0, 1].
        pmi1, pmi2, pmi3 = values[2], values[3], values[4]
        assert pmi1 <= pmi2 <= pmi3
        assert 0.0 < values[7] <= 1.0  # NPR1
        assert 0.0 < values[8] <= 1.0  # NPR2
        assert values[7] <= values[8]  # NPR1 <= NPR2 (Rodger's plane)

    def test_deterministic_across_calls(self) -> None:
        first = compute_3d_descriptors([ASPIRIN, ETHANOL], progress=False)
        second = compute_3d_descriptors([ASPIRIN, ETHANOL], progress=False)
        pd.testing.assert_frame_equal(
            first[list(D3D_COLUMNS)], second[list(D3D_COLUMNS)]
        )

    def test_single_conformer_works(self) -> None:
        frame = compute_3d_descriptors([ASPIRIN], n_conformers=1, progress=False)
        row = frame[list(D3D_COLUMNS)].iloc[0].to_numpy()
        assert np.isfinite(row).all()
        assert row[2] <= row[3] <= row[4]  # PMI ordering survives one conf

    def test_invalid_smiles_dropped_valid_rows_nan_free(self) -> None:
        frame = compute_3d_descriptors(
            [ASPIRIN, INVALID, ETHANOL], progress=False
        )
        assert frame.index.tolist() == [0, 2]  # the invalid row is gone
        assert frame["SMILES"].tolist() == [ASPIRIN, ETHANOL]
        assert np.isfinite(frame[list(D3D_COLUMNS)].to_numpy()).all()

    def test_failed_molecule_keeps_nan_row(self, monkeypatch) -> None:
        """A molecule whose 3D pipeline fails keeps its ROW as NaN.

        The notebook's failed-molecule behaviour: ``failed_indices``
        receives the index and the row stays (NaN cells), it is not
        dropped the way unparseable SMILES are.
        """
        from pharmaml.chem import descriptors as descriptors_module

        def broken(mol_h, n_conformers):  # pragma: no cover - stub
            raise ValueError("embedding exploded")

        monkeypatch.setattr(descriptors_module, "_shape_vector", broken)
        frame = compute_3d_descriptors([ASPIRIN, ETHANOL], progress=False)
        assert frame.shape == (2, 10)  # both rows kept
        assert frame[list(D3D_COLUMNS)].isna().all().all()

    def test_uff_fallback_for_mmff_untypable_molecule(self) -> None:
        """MMFF typing fails for selenium atoms (flag -1 results) -- the
        UFF fallback must take over and still produce finite values."""
        frame = compute_3d_descriptors(["C[Se]C"], progress=False)
        row = frame[list(D3D_COLUMNS)].iloc[0].to_numpy()
        assert np.isfinite(row).all()
        assert row[2] <= row[3] <= row[4]

    def test_bare_string_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            compute_3d_descriptors(ETHANOL, progress=False)

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="empty"):
            compute_3d_descriptors([], progress=False)

    def test_all_invalid_input_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="No valid SMILES"):
            compute_3d_descriptors([INVALID, "also bad"], progress=False)

    @pytest.mark.parametrize("value", [0, -1, 2.5, "3", True])
    def test_non_positive_or_non_int_conformers_rejected(self, value) -> None:
        with pytest.raises(DataValidationError, match="n_conformers"):
            compute_3d_descriptors([ETHANOL], n_conformers=value, progress=False)

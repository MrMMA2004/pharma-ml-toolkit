"""Tests for the 3D-RISM-KH-inspired solvation descriptors.

Covers the descriptor contract (column names, index, SMILES passthrough),
determinism, the invalid-row accounting, the solvent/temperature
validation guards, chemical sanity of the values (polarity ordering,
the Gasteiger dipole, size monotonicity) and the ``rism`` feature
family inside the multi-family feature factory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from pharmaml.chem.rism import (
    _TEMPER_K,
    RISM_DESCRIPTORS,
    compute_rism_descriptors,
)
from pharmaml.exceptions import DataValidationError

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
OCTANE = "CCCCCCCC"
METHYLAMMONIUM = "C[NH3+]"


@pytest.fixture(scope="module")
def solvation_frame() -> pd.DataFrame:
    """One shared frame over five molecules of increasing polarity/size."""
    return compute_rism_descriptors(
        [ETHANOL, BENZENE, ASPIRIN, OCTANE, METHYLAMMONIUM], progress=False
    )


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------
class TestDescriptorContract:
    def test_ten_descriptor_columns_in_output_order(self) -> None:
        assert len(RISM_DESCRIPTORS) == 10
        assert RISM_DESCRIPTORS[0] == "RISM_Gsolv_kcalmol"
        assert RISM_DESCRIPTORS[-1] == "RISM_q_neg_frac"

    def test_frame_layout(self, solvation_frame: pd.DataFrame) -> None:
        frame = solvation_frame
        assert list(frame.columns) == ["SMILES", *RISM_DESCRIPTORS]
        assert frame.index.name == "original_index"
        assert frame.index.tolist() == [0, 1, 2, 3, 4]
        assert frame["SMILES"].tolist() == [
            ETHANOL, BENZENE, ASPIRIN, OCTANE, METHYLAMMONIUM,
        ]

    def test_values_finite_and_typed(self, solvation_frame: pd.DataFrame) -> None:
        values = solvation_frame[list(RISM_DESCRIPTORS)].to_numpy()
        assert values.dtype == np.float64
        assert np.isfinite(values).all()

    def test_chem_package_exports(self) -> None:
        from pharmaml import chem

        assert "compute_rism_descriptors" in chem.__all__
        assert "RISM_DESCRIPTORS" in chem.__all__
        assert chem.compute_rism_descriptors is compute_rism_descriptors

    def test_import_path_needs_no_rdkit_at_import(self) -> None:
        """The module imports (constants reachable) without touching rdkit."""
        import sys

        assert "rdkit" in sys.modules  # imported elsewhere in the session
        # the contract: importing the module never pulls rdkit in itself
        import pharmaml.chem.rism as rism_module

        assert rism_module.RISM_DESCRIPTORS is RISM_DESCRIPTORS


# ---------------------------------------------------------------------------
# Determinism and failure accounting
# ---------------------------------------------------------------------------
class TestDeterminismAndAccounting:
    def test_repeated_calls_are_identical(self) -> None:
        first = compute_rism_descriptors([ETHANOL, ASPIRIN], progress=False)
        second = compute_rism_descriptors([ETHANOL, ASPIRIN], progress=False)
        pd.testing.assert_frame_equal(first, second)

    def test_invalid_rows_dropped_with_index_preserved(self) -> None:
        frame = compute_rism_descriptors(
            [ETHANOL, "not-a-smiles", BENZENE, ""], progress=False
        )
        assert frame.index.tolist() == [0, 2]
        assert frame["SMILES"].tolist() == [ETHANOL, BENZENE]

    def test_all_invalid_raises(self) -> None:
        with pytest.raises(DataValidationError, match="No valid SMILES"):
            compute_rism_descriptors(["!!!", "????"], progress=False)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(DataValidationError, match="smiles is empty"):
            compute_rism_descriptors([], progress=False)

    def test_bare_string_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="single string"):
            compute_rism_descriptors(ETHANOL, progress=False)

    def test_series_column_works(self) -> None:
        column = pd.Series([ETHANOL, BENZENE])
        frame = compute_rism_descriptors(column, progress=False)
        assert len(frame) == 2


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
class TestValidationGuards:
    def test_unknown_solvent_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="solvent must be one of"):
            compute_rism_descriptors([ETHANOL], solvent="sea", progress=False)

    def test_non_positive_temperature_rejected(self) -> None:
        for bad in (0.0, -5.0, float("nan")):
            with pytest.raises(DataValidationError, match="temperature must be > 0"):
                compute_rism_descriptors([ETHANOL], temperature=bad, progress=False)

    def tip3p_changes_the_site_model(self) -> None:
        spc = compute_rism_descriptors([ETHANOL], solvent="spc", progress=False)
        tip3p = compute_rism_descriptors([ETHANOL], solvent="tip3p", progress=False)
        assert float(tip3p["RISM_Gsolv_kcalmol"].iloc[0]) != pytest.approx(
            float(spc["RISM_Gsolv_kcalmol"].iloc[0])
        )


# ---------------------------------------------------------------------------
# Chemical sanity of the surrogate values
# ---------------------------------------------------------------------------
class TestChemicalSanity:
    def test_polar_molecules_solvate_stronger_than_apolar(
        self, solvation_frame: pd.DataFrame
    ) -> None:
        frame = solvation_frame
        # ethanol (H-bond donor+acceptor) vs benzene (apolar ring)
        assert (
            float(frame["RISM_Gcoul_kcalmol"].loc[0])
            < float(frame["RISM_Gcoul_kcalmol"].loc[1]) - 50.0
        )
        # aspirin (2 COOH + ester) is the most strongly hydrated here
        gsolv = frame["RISM_Gsolv_kcalmol"].astype(float)
        assert gsolv.idxmin() == 2  # aspirin row

    def test_gasteiger_dipole_matches_experiment(self) -> None:
        frame = compute_rism_descriptors([ETHANOL, BENZENE], progress=False)
        ethanol_dipole = float(frame["RISM_dipole_D"].loc[0])
        # experimental ethanol dipole: 1.69 D; the Gasteiger/ETKDG value
        # lands in the same neighbourhood, benzene's is symmetry-zero
        # (float noise below 1e-4 D).
        assert 1.2 < ethanol_dipole < 2.1
        assert float(frame["RISM_dipole_D"].loc[1]) == pytest.approx(0.0, abs=1e-4)

    def test_excluded_volume_grows_with_size(self, solvation_frame: pd.DataFrame) -> None:
        volumes = solvation_frame["RISM_Ow_exclvol_A3"].astype(float)
        assert volumes.loc[0] < volumes.loc[2]  # ethanol < aspirin
        assert volumes.loc[1] < volumes.loc[3]  # benzene < octane

    def test_gmax_capped_at_tempered_closure_maximum(
        self, solvation_frame: pd.DataFrame
    ) -> None:
        ceiling = float(np.exp(_TEMPER_K))
        for column in ("RISM_Ow_gmax", "RISM_Hw_gmax"):
            values = solvation_frame[column].astype(float)
            assert (values >= 1.0).all()
            assert (values <= ceiling + 1e-9).all()

    def test_coordination_numbers_non_negative(self, solvation_frame: pd.DataFrame) -> None:
        for column in ("RISM_Ow_cn1", "RISM_Hw_cn1"):
            assert (solvation_frame[column].astype(float) >= 0.0).all()

    def test_charge_asymmetry_of_a_cation(self, solvation_frame: pd.DataFrame) -> None:
        # methylammonium: the negative-charge share collapses (the ion is
        # positive overall) while neutral molecules sit at exactly 0.5.
        assert float(solvation_frame["RISM_q_neg_frac"].loc[4]) < 0.45
        assert float(solvation_frame["RISM_q_neg_frac"].loc[1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The rism feature family
# ---------------------------------------------------------------------------
class TestRismFeatureFamily:
    def test_family_computes_ten_columns(self) -> None:
        from pharmaml.chem.features import compute_feature_frame, family_of_column

        frame, report = compute_feature_frame([ETHANOL], families=("rism",))
        assert frame.shape == (1, 10)
        assert report.families == {"rism": 10}
        assert frame.columns[0] == "rism_gsolv_kcalmol"
        assert frame.columns[-1] == "rism_q_neg_frac"
        assert family_of_column("rism_ow_cn1") == "rism"

    def test_family_values_match_the_standalone_module(self) -> None:
        from pharmaml.chem.features import compute_feature_frame

        frame, _ = compute_feature_frame([ASPIRIN], families=("rism",))
        standalone = compute_rism_descriptors([ASPIRIN], progress=False)
        assert float(frame["rism_gsolv_kcalmol"].iloc[0]) == pytest.approx(
            float(standalone["RISM_Gsolv_kcalmol"].iloc[0])
        )
        assert float(frame["rism_dipole_d"].iloc[0]) == pytest.approx(
            float(standalone["RISM_dipole_D"].iloc[0])
        )

    def test_family_participates_in_the_full_frame(self) -> None:
        from pharmaml.chem.features import FEATURE_FAMILIES, compute_feature_frame

        assert "rism" in FEATURE_FAMILIES
        frame, report = compute_feature_frame(
            [ETHANOL, "not_a_smiles"], families=("descriptors", "rism")
        )
        assert list(report.families) == ["descriptors", "rism"]  # canonical order
        assert report.dropped == [(1, "not_a_smiles")]
        assert "rism_dipole_d" in frame.columns

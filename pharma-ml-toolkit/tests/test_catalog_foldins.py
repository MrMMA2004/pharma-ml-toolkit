"""Tests for the 140-repository catalog's remaining offline fold-ins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.docking.dockstring import (
    DOCKSTRING_TARGETS,
    evaluate_objective,
    f2_objective,
    get_target,
    parse_search_box,
    promiscuous_ppar_objective,
    qed_penalty,
    selective_jak2_objective,
    target_frame,
)
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.dta import (
    AMINO_ACIDS,
    DEEPDTA_CHARCANSMISET,
    DEEPDTA_CHARISOSMISET,
    DEEPDTA_CHARPROTSET,
    deepdta_rm2,
    demo_dta_frame,
    label_sequence,
    label_smiles,
    one_hot_sequence,
    one_hot_smiles,
    paired_dta_features,
    protein_feature_frame,
)
from pharmaml.pkpd.pregnancy import (
    PREGNANCY_MODELS,
    MaternalFetalParameters,
    fertilization_age_weeks,
    pk_sim_age_years,
    pregnancy_model_frame,
    simulate_maternal_fetal,
)


class TestDeepDTAEncoding:
    def test_verbatim_alphabet_and_padding(self) -> None:
        assert len(DEEPDTA_CHARPROTSET) == 25
        assert set(DEEPDTA_CHARPROTSET.values()) == set(range(1, 26))
        assert label_sequence("ACX", max_length=5).tolist() == [1, 2, 24, 0, 0]
        assert label_sequence("AC", max_length=1).tolist() == [1]

    def test_one_hot_is_left_aligned_and_zero_padded(self) -> None:
        encoded = one_hot_sequence("AC", max_length=4)
        assert encoded.shape == (4, 25)
        assert encoded[0, 0] == encoded[1, 1] == 1.0
        assert np.count_nonzero(encoded) == 2
        assert np.count_nonzero(encoded[2:]) == 0

    def test_verbatim_smiles_alphabets_and_padding(self) -> None:
        assert len(DEEPDTA_CHARCANSMISET) == 62
        assert len(DEEPDTA_CHARISOSMISET) == 64
        assert set(DEEPDTA_CHARCANSMISET.values()) == set(range(1, 63))
        assert set(DEEPDTA_CHARISOSMISET.values()) == set(range(1, 65))
        assert label_smiles("C(C)O", max_length=6).tolist() == [42, 1, 42, 31, 48, 0]
        assert label_smiles("C(C)O", max_length=6, isomeric=False).tolist() == [20, 4, 20, 3, 31, 0]
        assert label_smiles("CCO", max_length=2).tolist() == [42, 42]

    def test_smiles_one_hot_and_validation(self) -> None:
        encoded = one_hot_smiles("CO", max_length=4, isomeric=False)
        assert encoded.shape == (4, 62)
        assert encoded[0, 19] == encoded[1, 30] == 1.0
        assert np.count_nonzero(encoded) == 2
        with pytest.raises(DataValidationError, match="outside the selected DeepDTA"):
            label_smiles("C@O", isomeric=False)
        with pytest.raises(DataValidationError, match="non-empty"):
            one_hot_smiles("")

    def test_sequence_validation_is_explicit(self) -> None:
        with pytest.raises(DataValidationError, match="outside DeepDTA"):
            label_sequence("ACJ")
        with pytest.raises(DataValidationError, match="empty"):
            label_sequence(" \n")
        with pytest.raises(DataValidationError, match="positive integer"):
            label_sequence("AC", max_length=0)
        with pytest.raises(DataValidationError, match="bare string"):
            protein_feature_frame("ACDE")

    def test_amino_acid_and_dipeptide_composition(self) -> None:
        frame = protein_feature_frame(["AAX", "ACDE"], dipeptides=True)
        assert len(AMINO_ACIDS) == 20
        assert frame.shape == (2, 424)  # 3 audits + 20 AAC + 1 audit + 400 pairs
        assert frame.loc[0, "prot_aac_A"] == pytest.approx(2 / 3)
        assert frame.loc[0, "prot_nonstandard_fraction"] == pytest.approx(1 / 3)
        assert frame.loc[0, "prot_dipep_AA"] == pytest.approx(0.5)
        assert "prot_dipep_AX" not in frame.columns  # X is not a standard residue
        assert frame.loc[1, "prot_dipep_AC"] == pytest.approx(1 / 3)
        assert frame.loc[0, "prot_dipeptide_standard_fraction"] == pytest.approx(0.5)


class TestPairedDTAFeatures:
    def test_invalid_ligand_drops_the_whole_pair(self) -> None:
        pairs = pd.DataFrame(
            {
                "smiles": ["CCO", "broken", "CCN"],
                "protein": ["ACDE", "AAAA", "ACDF"],
                "affinity": [5.0, 99.0, 7.0],
            }
        )
        result = paired_dta_features(pairs)
        assert result.kept_indices == (0, 2)
        assert result.dropped == ((1, "broken"),)
        assert result.y is not None and result.y.tolist() == [5.0, 7.0]
        assert result.n_ligand_features == 2048
        assert result.n_protein_features == 23
        assert result.training_frame().shape == (2, 2072)

    def test_bad_affinity_cannot_be_silently_dropped(self) -> None:
        pairs = demo_dta_frame().head(2).astype({"affinity": object})
        pairs.loc[1, "affinity"] = "unknown"
        with pytest.raises(DataValidationError, match="source rows: 1"):
            paired_dta_features(pairs)

    def test_missing_columns_are_named(self) -> None:
        with pytest.raises(DataValidationError, match="protein"):
            paired_dta_features(pd.DataFrame({"smiles": ["CCO"]}))

    def test_demo_frame_is_explicitly_synthetic_and_usable(self) -> None:
        demo = demo_dta_frame()
        assert demo.shape == (12, 3)
        result = paired_dta_features(demo, dipeptides=True)
        assert result.X.shape == (12, 2472)


class TestDeepDTAMetric:
    def test_perfect_and_rescaled_predictions(self) -> None:
        assert deepdta_rm2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert deepdta_rm2([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_intercept_shift_is_penalised(self) -> None:
        assert deepdta_rm2([1, 2, 3], [2, 3, 4]) < 1.0

    def test_independent_formula_pin(self) -> None:
        observed = np.array([1.0, 2.0, 4.0, 8.0])
        predicted = np.array([1.1, 1.8, 4.4, 7.2])
        r2 = np.corrcoef(observed, predicted)[0, 1] ** 2
        slope = np.sum(observed * predicted) / np.sum(predicted**2)
        r0_squared = 1 - np.sum((observed - slope * predicted) ** 2) / np.sum(
            (observed - observed.mean()) ** 2
        )
        expected = r2 * (1 - np.sqrt(abs(r2**2 - r0_squared**2)))
        assert deepdta_rm2(observed, predicted) == pytest.approx(expected)

    def test_degenerate_metric_input_is_refused(self) -> None:
        with pytest.raises(DataValidationError, match="undefined"):
            deepdta_rm2([1, 1], [1, 2])


class TestPregnancyRegistryAndAge:
    def test_all_distributed_compound_directories_are_indexed(self) -> None:
        assert len(PREGNANCY_MODELS) == 16
        assert "Acetaminophen" in PREGNANCY_MODELS
        assert "Amoxicillin" not in PREGNANCY_MODELS  # upstream links a separate repository
        frame = pregnancy_model_frame()
        assert len(frame) == 16
        assert frame["scope"].str.contains("not a fitted").all()

    def test_pk_sim_fertilization_age_round_trip(self) -> None:
        assert pk_sim_age_years(0) == 30.0
        assert fertilization_age_weeks(pk_sim_age_years(38)) == pytest.approx(38.0)
        with pytest.raises(DataValidationError):
            pk_sim_age_years(43)
        with pytest.raises(DataValidationError):
            fertilization_age_weeks(29.9)


class TestMaternalFetalModel:
    @staticmethod
    def params(**changes: float) -> MaternalFetalParameters:
        values = {
            "maternal_volume_l": 50.0,
            "fetal_volume_l": 3.0,
            "maternal_clearance_l_h": 0.0,
            "fetal_clearance_l_h": 0.0,
            "placental_clearance_l_h": 2.0,
            "maternal_fu": 0.8,
            "fetal_fu": 0.4,
        }
        values.update(changes)
        return MaternalFetalParameters(**values)

    def test_zero_transfer_means_zero_fetal_exposure(self) -> None:
        result = simulate_maternal_fetal(
            self.params(placental_clearance_l_h=0.0), dose_mg=100, times=[0, 1, 2]
        )
        assert result.frame["fetal_amount_mg"].max() == 0.0
        assert result.fetal_maternal_auc_ratio == 0.0

    def test_amount_is_conserved_without_elimination(self) -> None:
        result = simulate_maternal_fetal(self.params(), dose_mg=100, times=np.linspace(0, 12, 25))
        total = result.frame["maternal_amount_mg"] + result.frame["fetal_amount_mg"]
        assert np.allclose(total, 100.0, atol=1e-6)

    def test_equilibrium_uses_separate_unbound_fractions(self) -> None:
        result = simulate_maternal_fetal(self.params(), dose_mg=100, times=[0, 50, 100])
        final = result.frame.iloc[-1]
        assert final["maternal_unbound_mg_l"] == pytest.approx(
            final["fetal_unbound_mg_l"], rel=1e-5
        )
        # Same unbound concentration, different binding: total concentrations differ.
        assert final["maternal_concentration_mg_l"] < final["fetal_concentration_mg_l"]

    def test_maternal_clearance_removes_amount(self) -> None:
        result = simulate_maternal_fetal(
            self.params(maternal_clearance_l_h=1.0, placental_clearance_l_h=0.0),
            dose_mg=100,
            times=[0, 1],
        )
        assert result.frame.iloc[-1]["maternal_amount_mg"] == pytest.approx(
            100 * np.exp(-1 / 50), rel=1e-5
        )

    def test_invalid_parameters_and_times_are_refused(self) -> None:
        with pytest.raises(DataValidationError, match="maternal_fu"):
            simulate_maternal_fetal(self.params(maternal_fu=0), dose_mg=1, times=[0, 1])
        with pytest.raises(DataValidationError, match="start at 0"):
            simulate_maternal_fetal(self.params(), dose_mg=1, times=[1, 2])
        with pytest.raises(DataValidationError, match="strictly increasing"):
            simulate_maternal_fetal(self.params(), dose_mg=1, times=[0, 0])


class TestDockstringRegistry:
    def test_all_58_upstream_boxes_round_trip(self) -> None:
        assert len(DOCKSTRING_TARGETS) == 58
        assert len(target_frame()) == 58
        for target in DOCKSTRING_TARGETS.values():
            parsed = parse_search_box(target.config_text())
            assert parsed.center == target.center
            assert parsed.size == target.size
        assert get_target("DRD2").size == (30.0, 30.0, 32.0)

    def test_unknown_target_and_bad_box_are_refused(self) -> None:
        with pytest.raises(DataValidationError, match="unknown DOCKSTRING"):
            get_target("NOT_A_TARGET")
        with pytest.raises(DataValidationError, match="missing"):
            parse_search_box("center_x = 0")
        with pytest.raises(DataValidationError, match="positive"):
            parse_search_box(get_target("F2").config_text().replace("size_x = 30.000", "size_x = 0"))


class TestDockstringObjectives:
    def test_qed_and_f2(self) -> None:
        assert qed_penalty(0.8) == pytest.approx(2.0)
        assert f2_objective(f2=-9.0, qed=0.8) == pytest.approx(-7.0)
        assert evaluate_objective("F2", {"F2": -9.0}, qed=0.8) == pytest.approx(-7.0)

    def test_promiscuous_ppar_and_nan_contract(self) -> None:
        assert promiscuous_ppar_objective(ppara=-9, ppard=-8, pparg=-10, qed=1) == -8
        assert np.isnan(
            promiscuous_ppar_objective(ppara=float("nan"), ppard=-8, pparg=-10, qed=1)
        )

    def test_corrected_selective_jak2_formula(self) -> None:
        # LCK better than its -8.1 median incurs a selectivity penalty.
        assert selective_jak2_objective(jak2=-9, lck=-9, qed=0.8) == pytest.approx(-6.1)
        assert selective_jak2_objective(jak2=-9, lck=-7, qed=0.8) == pytest.approx(-7.0)
        assert evaluate_objective(
            "selective_JAK2", {"JAK2": -9, "LCK": -9}, qed=0.8
        ) == pytest.approx(-6.1)

    def test_required_scores_and_qed_are_validated(self) -> None:
        with pytest.raises(DataValidationError, match="LCK"):
            evaluate_objective("selective_JAK2", {"JAK2": -9}, qed=0.8)
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            qed_penalty(1.1)

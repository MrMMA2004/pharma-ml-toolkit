"""Unit tests for dosage-regimen calculations."""

from __future__ import annotations

import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma.dosage import (
    bsa_mosteller,
    clarks_rule,
    dose_by_bsa,
    dose_by_weight,
    infusion_rate,
    loading_dose,
    maintenance_dose,
    renal_dose_adjustment,
    tdm_trough_adjustment,
    youngs_rule,
)


class TestRegimenDesign:
    def test_maintenance_dose(self) -> None:
        # CL=5 L/h, Css=10, tau=12 h, F=1 -> 600 mg.
        assert maintenance_dose(5.0, 10.0, 12.0) == pytest.approx(600.0)

    def test_maintenance_dose_with_bioavailability(self) -> None:
        assert maintenance_dose(5.0, 10.0, 12.0, f=0.5) == pytest.approx(1200.0)

    def test_maintenance_rejects_f_above_one(self) -> None:
        with pytest.raises(DataValidationError, match="bioavailability"):
            maintenance_dose(5.0, 10.0, 12.0, f=1.5)

    def test_loading_dose(self) -> None:
        assert loading_dose(40.0, 15.0) == pytest.approx(600.0)
        assert loading_dose(40.0, 15.0, f=0.8) == pytest.approx(750.0)

    def test_infusion_rate(self) -> None:
        assert infusion_rate(4.0, 3.0) == pytest.approx(12.0)


class TestBodySizeScaling:
    def test_bsa_mosteller_reference(self) -> None:
        # sqrt(170*70/3600) = 1.8185
        assert bsa_mosteller(170, 70) == pytest.approx(1.8185, rel=1e-3)

    def test_dose_by_weight(self) -> None:
        assert dose_by_weight(15.0, 60.0) == pytest.approx(900.0)

    def test_dose_by_weight_ceiling_raises(self) -> None:
        with pytest.raises(DataValidationError, match="exceeds"):
            dose_by_weight(15.0, 60.0, max_dose=500.0)

    def test_dose_by_bsa(self) -> None:
        dose = dose_by_bsa(100.0, 170, 70)
        assert dose == pytest.approx(100.0 * bsa_mosteller(170, 70))

    def test_implausible_measurements_rejected(self) -> None:
        with pytest.raises(DataValidationError):
            bsa_mosteller(10, 70)
        with pytest.raises(DataValidationError):
            bsa_mosteller(170, 1000)


class TestLegacyPaediatricRules:
    def test_youngs_rule(self) -> None:
        # age 6, adult 500 -> 500*6/18 = 166.67
        assert youngs_rule(6, 500.0) == pytest.approx(166.6667, rel=1e-4)

    def test_youngs_rule_age_bounds(self) -> None:
        with pytest.raises(DataValidationError):
            youngs_rule(0.5, 500.0)
        with pytest.raises(DataValidationError):
            youngs_rule(15, 500.0)

    def test_clarks_rule(self) -> None:
        assert clarks_rule(75.0, 300.0) == pytest.approx(150.0)


class TestRenalAdjustment:
    def test_half_function_halves_dose(self) -> None:
        assert renal_dose_adjustment(200.0, 50.0, 100.0) == pytest.approx(100.0)

    def test_floor_applied_for_anuria(self) -> None:
        # CrCl=0 with floor 0.25 -> 25% of dose.
        assert renal_dose_adjustment(200.0, 0.0, 100.0, floor_fraction=0.25) == pytest.approx(50.0)

    def test_invalid_floor(self) -> None:
        with pytest.raises(DataValidationError):
            renal_dose_adjustment(200.0, 50.0, 100.0, floor_fraction=1.5)


class TestTDM:
    def test_proportional_adjustment(self) -> None:
        # 500 mg at trough 8 -> trough 12 needs 750 mg.
        assert tdm_trough_adjustment(500.0, 8.0, 12.0) == pytest.approx(750.0)

    def test_reduction(self) -> None:
        assert tdm_trough_adjustment(500.0, 20.0, 10.0) == pytest.approx(250.0)

    def test_min_dose_floor(self) -> None:
        assert tdm_trough_adjustment(500.0, 30.0, 10.0, min_dose=250.0) == pytest.approx(250.0)

    def test_extreme_ratio_refused(self) -> None:
        with pytest.raises(DataValidationError, match="non-linear"):
            tdm_trough_adjustment(100.0, 1.0, 12.0)  # ratio 12 > 5

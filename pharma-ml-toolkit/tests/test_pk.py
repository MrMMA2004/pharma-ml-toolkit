"""Unit tests for pharmacokinetic calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma import pk
from pharmaml.pharma.pk import (
    accumulation_ratio,
    auc_trapezoid,
    aucm_trapezoid,
    bioavailability,
    clearance,
    creatinine_clearance,
    elimination_rate_constant,
    half_life,
    iv_bolus_concentration,
    iv_infusion_steady_state,
    mrt,
    nca_summary,
    oral_cmax,
    oral_tmax,
    vd_from_dose_auc,
    volume_of_distribution,
)


class TestHalfLifeRoundtrip:
    def test_roundtrip(self) -> None:
        for t_half in (0.5, 2.0, 6.0, 24.0):
            assert half_life(elimination_rate_constant(t_half)) == pytest.approx(t_half)

    def test_known_value(self) -> None:
        assert half_life(0.35) == pytest.approx(np.log(2) / 0.35)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(DataValidationError):
            elimination_rate_constant(0.0)
        with pytest.raises(DataValidationError):
            half_life(-1.0)


class TestIVBolus:
    def test_decay(self) -> None:
        assert iv_bolus_concentration(10.0, 0.5, 1.0) == pytest.approx(10.0 * np.exp(-0.5))

    def test_array_time(self) -> None:
        result = iv_bolus_concentration(10.0, 0.5, np.array([0.0, 1.0, 2.0]))
        assert result[0] == pytest.approx(10.0)
        assert result[2] == pytest.approx(10.0 * np.exp(-1.0))


class TestOralOneCompartment:
    def test_tmax_formula(self) -> None:
        # ka=1.2, ke=0.35 -> ln(1.2/0.35)/(1.2-0.35)
        assert oral_tmax(1.2, 0.35) == pytest.approx(np.log(1.2 / 0.35) / 0.85)

    def test_tmax_rejects_flip_flop(self) -> None:
        with pytest.raises(DataValidationError):
            oral_tmax(ka=0.2, ke=0.5)

    def test_cmax_matches_numerical_peak(self) -> None:
        ka, ke, vd, dose, f = 1.2, 0.35, 35.0, 250.0, 0.9
        predicted = oral_cmax(dose, ka, ke, vd, f)
        times = np.linspace(0, 6, 6001)
        coefficient = f * dose * ka / (vd * (ka - ke))
        conc = coefficient * (np.exp(-ke * times) - np.exp(-ka * times))
        assert predicted == pytest.approx(conc.max(), rel=1e-3)


class TestAUC:
    def test_constant_concentration(self) -> None:
        # AUC of C=5 over [0, 10] is exactly 50.
        assert auc_trapezoid([0, 5, 10], [5, 5, 5]) == pytest.approx(50.0)

    def test_linear_ramp(self) -> None:
        # C(t) = t on [0, 4]: AUC = 8 analytically.
        assert auc_trapezoid([0, 2, 4], [0, 2, 4]) == pytest.approx(8.0)

    def test_exponential_log_vs_linear(self) -> None:
        # For pure exponential decay the log trapezoid is exact:
        # AUC of e^-t over [0, 3] = 1 - e^-3.
        times = np.linspace(0, 3, 31)
        conc = np.exp(-times)
        exact = 1 - np.exp(-3)
        log_auc = auc_trapezoid(times, conc, method="log")
        lin_auc = auc_trapezoid(times, conc, method="linear")
        assert abs(log_auc - exact) < abs(lin_auc - exact)

    def test_rejects_non_ascending_times(self) -> None:
        with pytest.raises(DataValidationError, match="ascending"):
            auc_trapezoid([0, 2, 2], [1, 1, 1])

    def test_rejects_negative_conc(self) -> None:
        with pytest.raises(DataValidationError):
            auc_trapezoid([0, 1], [1.0, -0.5])

    def test_rejects_log_with_zero_conc(self) -> None:
        with pytest.raises(DataValidationError):
            auc_trapezoid([0, 1, 2], [1.0, 0.0, 1.0], method="log")

    def test_bad_method(self) -> None:
        with pytest.raises(DataValidationError):
            auc_trapezoid([0, 1], [1.0, 2.0], method="cubic")

    def test_length_mismatch(self) -> None:
        with pytest.raises(DataValidationError):
            auc_trapezoid([0, 1, 2], [1.0, 2.0])


class TestMoments:
    def test_aucm_known(self) -> None:
        # AUMC of constant C=5 over [0, 10] = 5 * t^2/2 |0..10 = 250.
        assert aucm_trapezoid([0, 5, 10], [5, 5, 5]) == pytest.approx(250.0)

    def test_mrt_identity(self) -> None:
        # Constant infusion-like profile: MRT = midpoint-ish; check ratio.
        assert mrt([0, 5, 10], [5, 5, 5]) == pytest.approx(250.0 / 50.0)


class TestDerivedParameters:
    def test_clearance(self) -> None:
        assert clearance(dose=500.0, auc=50.0) == pytest.approx(10.0)
        assert clearance(dose=500.0, auc=50.0, f=0.5) == pytest.approx(5.0)

    def test_vd(self) -> None:
        assert volume_of_distribution(10.0, 0.5) == pytest.approx(20.0)

    def test_bioavailability_symmetry(self) -> None:
        # Identical doses and AUCs -> F = 1.
        assert bioavailability(100.0, 20.0, 100.0, 20.0) == pytest.approx(1.0)
        assert bioavailability(100.0, 10.0, 100.0, 20.0) == pytest.approx(0.5)

    def test_accumulation_ratio_half_life_interval(self) -> None:
        # Dosing every t1/2 -> R = 2.
        assert accumulation_ratio(tau=6.0, t_half=6.0) == pytest.approx(2.0)

    def test_accumulation_ratio_long_interval(self) -> None:
        assert accumulation_ratio(tau=1000.0, t_half=1.0) == pytest.approx(1.0, abs=1e-6)


class TestCockcroftGault:
    def test_reference_male(self) -> None:
        # (140-40)*80/(72*1.0) = 111.11
        assert creatinine_clearance(40, 80, 1.0, "male") == pytest.approx(111.11, rel=1e-3)

    def test_female_factor(self) -> None:
        male = creatinine_clearance(40, 80, 1.0, "male")
        female = creatinine_clearance(40, 80, 1.0, "female")
        assert female == pytest.approx(0.85 * male)

    def test_rejects_bad_sex(self) -> None:
        with pytest.raises(DataValidationError):
            creatinine_clearance(40, 80, 1.0, "other")

    def test_rejects_bad_age(self) -> None:
        with pytest.raises(DataValidationError):
            creatinine_clearance(150, 80, 1.0, "male")

    def test_rejects_non_string_sex(self) -> None:
        # Regression: a non-string sex used to escape as a raw
        # AttributeError ('NoneType' has no attribute 'strip').
        with pytest.raises(DataValidationError, match="sex"):
            creatinine_clearance(40, 80, 1.0, None)
        with pytest.raises(DataValidationError, match="sex"):
            creatinine_clearance(40, 80, 1.0, 123)


class TestNCASummary:
    def test_synthetic_profile(self) -> None:
        times = np.linspace(0, 24, 49)
        ka, ke, f, dose, vd = 1.2, 0.35, 0.9, 250.0, 35.0
        coefficient = f * dose * ka / (vd * (ka - ke))
        conc = coefficient * (np.exp(-ke * times) - np.exp(-ka * times))
        summary = nca_summary(times, conc)

        assert summary.c_max == pytest.approx(conc.max())
        assert summary.t_max == pytest.approx(times[np.argmax(conc)])
        assert summary.auc > 0
        # Terminal half-life should recover ~ln2/0.35 = 1.98 h (the tail
        # is ke-dominated for ka >> ke).
        assert summary.half_life_terminal == pytest.approx(np.log(2) / ke, rel=0.2)
        assert summary.as_dict()["AUC"] == summary.auc

    def test_all_zero_conc_handles_gracefully(self) -> None:
        # AUC -> 0 but Cmax/Tmax still defined; MRT may be inf.
        summary = nca_summary([0, 1, 2], [0.0, 0.0, 0.0])
        assert summary.c_max == 0.0
        assert summary.auc == 0.0


def test_module_docstring_exports() -> None:
    """The pk module re-exports its public API through pharmaml.pharma."""
    assert callable(pk.auc_trapezoid)
    assert callable(pk.nca_summary)


class TestVdFromDoseAuc:
    def test_plain_ratio_is_clearance_not_volume(self) -> None:
        # Dose/AUC = CL for an IV dose (F = 1) -- the honest framing
        assert vd_from_dose_auc(100.0, 20.0) == pytest.approx(5.0)

    def test_mrt_gives_steady_state_volume(self) -> None:
        # Vss = Dose*MRT/AUC (statistical moments)
        assert vd_from_dose_auc(100.0, 20.0, mrt=3.0) == pytest.approx(15.0)

    def test_agrees_with_volume_of_distribution(self) -> None:
        # One-compartment IV bolus: MRT = 1/ke, so Vss == Vd == CL/ke
        dose, auc, ke = 100.0, 20.0, 0.5
        cl = clearance(dose, auc)          # 5.0
        vss = vd_from_dose_auc(dose, auc, mrt=1.0 / ke)
        assert vss == pytest.approx(volume_of_distribution(cl, ke))

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(DataValidationError):
            vd_from_dose_auc(-1.0, 2.0)
        with pytest.raises(DataValidationError):
            vd_from_dose_auc(1.0, 0.0)
        with pytest.raises(DataValidationError):
            vd_from_dose_auc(1.0, 2.0, mrt=0.0)


def test_pharma_package_surface() -> None:
    """`from pharmaml.pharma import X` works for every public symbol."""
    import pharmaml.pharma as pharma

    expected = {
        # PK
        "PKSummary", "nca_summary", "validate_concentration_series",
        "iv_infusion_steady_state", "vd_from_dose_auc", "mrt",
        # PD
        "linear_effect", "ec50_for_target_effect", "effect_table",
        # dosage
        "clarks_rule", "youngs_rule", "infusion_rate",
        "renal_dose_adjustment", "tdm_trough_adjustment",
    }
    missing = [name for name in expected if not hasattr(pharma, name)]
    assert not missing, f"pharmaml.pharma is missing: {missing}"
    # star-import parity: __all__ lists everything importable
    for name in expected:
        assert name in pharma.__all__, name


class TestLogTrapezoid:
    """The ``method='log'`` contract: Purves' logarithmic trapezoid."""

    def test_exact_for_exponential_segment(self) -> None:
        # Regression: a 10x decay over one interval must integrate to
        # (c1 - c2)/ln(c1/c2) = 9/ln(10) = 3.9089.  The old geometric-mean
        # implementation returned sqrt(10) = 3.1623 (-19%).
        assert auc_trapezoid([0.0, 1.0], [10.0, 1.0], method="log") == pytest.approx(
            9.0 / math.log(10)
        )

    def test_exact_for_two_decade_decay(self) -> None:
        # Regression: 100x decay -- the old implementation was off by -53%.
        assert auc_trapezoid([0.0, 1.0], [100.0, 1.0], method="log") == pytest.approx(
            99.0 / math.log(100)
        )

    def test_mixed_flat_and_decaying_segments(self) -> None:
        # Flat segments (c1 == c2) are the log rule's 0/0 corner and fall
        # back to the linear rule; decaying segments use the log rule.
        expected = 5.0 + 5.0 + 4.5 / math.log(10)
        assert auc_trapezoid(
            [0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 0.5], method="log"
        ) == pytest.approx(expected)

    def test_flat_profile_log_does_not_crash(self) -> None:
        # PINNING (already correct before this round): constant profiles
        # integrate exactly under either rule.
        assert auc_trapezoid(
            [0.0, 1.0, 2.0], [5.0, 5.0, 5.0], method="log"
        ) == pytest.approx(10.0)

    def test_ascending_segment_uses_log_rule(self) -> None:
        # The log rule is direction-symmetric: 1 -> 10 over one interval.
        assert auc_trapezoid([0.0, 1.0], [1.0, 10.0], method="log") == pytest.approx(
            9.0 / math.log(10)
        )


class TestIVBolusSequenceTimes:
    """``iv_bolus_concentration`` accepts sequences, not just ndarrays."""

    def test_list_and_tuple_times(self) -> None:
        # Regression: list/tuple time input crashed with a raw TypeError.
        expected = 10.0 * np.exp(-0.5 * np.array([0.0, 1.0, 2.0]))
        for t in ([0.0, 1.0, 2.0], (0.0, 1.0, 2.0)):
            result = iv_bolus_concentration(10.0, 0.5, t)
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, expected)

    def test_series_times(self) -> None:
        # Regression: Series time input crashed with a raw TypeError.
        result = iv_bolus_concentration(10.0, 0.5, pd.Series([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            result, 10.0 * np.exp(-0.5 * np.array([0.0, 1.0, 2.0]))
        )

    def test_negative_time_in_sequence_raises(self) -> None:
        # PINNING (already correct before this round).
        with pytest.raises(DataValidationError, match="non-negative"):
            iv_bolus_concentration(10.0, 0.5, [0.0, -1.0])


class TestSeriesInputValidation:
    """Non-numeric series values must raise the documented typed error."""

    def test_non_numeric_concentration_raises_typed_error(self) -> None:
        # Regression: junk text escaped as a raw numpy ValueError
        # ("could not convert string to float") instead of DataValidationError.
        with pytest.raises(DataValidationError, match="numeric"):
            auc_trapezoid([0.0, 1.0], ["low", "high"])

    def test_non_numeric_times_raise_typed_error(self) -> None:
        with pytest.raises(DataValidationError, match="numeric"):
            auc_trapezoid(["now", "later"], [1.0, 2.0])


class TestNCATerminalSlope:
    def test_trailing_zero_samples_do_not_poison_terminal_slope(self) -> None:
        # PINNING (already correct before this round): BLQ zeros at the tail
        # are excluded -- the slope comes from the last two POSITIVE samples
        # (6 -> 3 over 1 h, i.e. t1/2 = 1 h).
        summary = nca_summary([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 6.0, 3.0, 0.0, 0.0])
        assert summary.half_life_terminal == pytest.approx(1.0)

    def test_rising_tail_gives_nan_half_life(self) -> None:
        # PINNING (already correct before this round): an ascending tail has
        # no terminal elimination phase.
        summary = nca_summary([0.0, 1.0, 2.0], [1.0, 2.0, 4.0])
        assert math.isnan(summary.half_life_terminal)


class TestIVInfusionSteadyState:
    def test_css_is_rate_over_clearance(self) -> None:
        # PINNING (first behavioural test for this public function -- it had
        # zero coverage beyond hasattr checks).
        assert iv_infusion_steady_state(50.0, 10.0) == pytest.approx(5.0)
        assert iv_infusion_steady_state(30.0, 2.0) == pytest.approx(15.0)

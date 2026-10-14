"""Unit tests for pharmacokinetic calculations."""

from __future__ import annotations

import numpy as np
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
    mrt,
    nca_summary,
    oral_cmax,
    oral_tmax,
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

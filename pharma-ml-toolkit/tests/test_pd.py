"""Unit tests for pharmacodynamic models."""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma.pd import (
    ec50_for_target_effect,
    effect_table,
    emax_effect,
    inhibitory_emax_effect,
    linear_effect,
    sigmoid_emax_effect,
)


class TestEmax:
    def test_half_max_at_ec50(self) -> None:
        assert emax_effect(10.0, e0=0.0, emax=100.0, ec50=10.0) == pytest.approx(50.0)

    def test_baseline_and_saturation(self) -> None:
        assert emax_effect(0.0, e0=5.0, emax=100.0, ec50=10.0) == pytest.approx(5.0)
        large = emax_effect(1e9, e0=0.0, emax=100.0, ec50=10.0)
        assert large == pytest.approx(100.0)

    def test_array_input(self) -> None:
        result = emax_effect(np.array([0.0, 10.0, 1e9]), 0.0, 100.0, 10.0)
        assert isinstance(result, np.ndarray)
        assert result[1] == pytest.approx(50.0)

    def test_rejects_negative_conc(self) -> None:
        with pytest.raises(DataValidationError):
            emax_effect(-1.0, 0.0, 100.0, 10.0)

    def test_rejects_bad_ec50(self) -> None:
        with pytest.raises(DataValidationError):
            emax_effect(1.0, 0.0, 100.0, ec50=0.0)


class TestSigmoidEmax:
    def test_hill_one_equals_hyperbolic(self) -> None:
        for conc in (1.0, 5.0, 25.0):
            simple = emax_effect(conc, 2.0, 80.0, 10.0)
            sigmoid = sigmoid_emax_effect(conc, 2.0, 80.0, 10.0, hill=1.0)
            assert simple == pytest.approx(sigmoid)

    def test_steeper_hill_passes_halfway_faster(self) -> None:
        below_ec50 = 8.0
        gentle = sigmoid_emax_effect(below_ec50, 0.0, 100.0, 10.0, hill=1.0)
        steep = sigmoid_emax_effect(below_ec50, 0.0, 100.0, 10.0, hill=3.0)
        assert steep < gentle  # below EC50 the steep curve is *lower*

    def test_half_max_at_ec50_any_hill(self) -> None:
        for hill in (0.5, 1.0, 2.0, 4.0):
            assert sigmoid_emax_effect(10.0, 0.0, 100.0, 10.0, hill) == pytest.approx(50.0)


class TestInhibitoryEmax:
    def test_baseline_at_zero(self) -> None:
        assert inhibitory_emax_effect(0.0, e0=90.0, imax=80.0, ic50=5.0) == pytest.approx(90.0)

    def test_half_inhibition_at_ic50(self) -> None:
        assert inhibitory_emax_effect(5.0, 90.0, 80.0, 5.0) == pytest.approx(90.0 - 40.0)

    def test_saturation_toward_e0_minus_imax(self) -> None:
        far = inhibitory_emax_effect(1e9, 90.0, 80.0, 5.0)
        assert far == pytest.approx(10.0)

    def test_hill_scaling(self) -> None:
        value = inhibitory_emax_effect(5.0, 90.0, 80.0, 5.0, hill=2.0)
        assert value == pytest.approx(50.0)  # 90 - 80*25/(25+25)


class TestLinear:
    def test_identity(self) -> None:
        assert linear_effect(4.0, slope=1.5, intercept=2.0) == pytest.approx(8.0)


class TestInverseEmax:
    def test_roundtrip(self) -> None:
        for conc in (2.0, 10.0, 40.0):
            effect = sigmoid_emax_effect(conc, 0.0, 100.0, 10.0, hill=1.5)
            recovered = ec50_for_target_effect(effect, ec50=10.0, e0=0.0, emax=100.0, hill=1.5)
            assert recovered == pytest.approx(conc, rel=1e-9)

    def test_unreachable_targets_raise(self) -> None:
        with pytest.raises(DataValidationError, match="unreachable"):
            ec50_for_target_effect(120.0, ec50=10.0, e0=0.0, emax=100.0)  # > Emax
        with pytest.raises(DataValidationError):
            ec50_for_target_effect(-5.0, ec50=10.0, e0=0.0, emax=100.0)  # < E0

    def test_zero_emax_raises_domain_error(self) -> None:
        # Regression: emax=0 used to escape as a raw ZeroDivisionError.
        with pytest.raises(DataValidationError, match="non-zero"):
            ec50_for_target_effect(5.0, ec50=10.0, e0=0.0, emax=0.0)


class TestEffectTable:
    def test_parallel_lists(self) -> None:
        grid = [0.0, 1.0, 10.0, 100.0]
        table = effect_table(grid, e0=0.0, emax=100.0, ec50=10.0, hill=1.0)
        assert table["concentration"] == grid
        assert len(table["effect"]) == len(grid)
        assert table["effect"][2] == pytest.approx(50.0)

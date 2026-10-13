"""Unit tests for unit conversion and molar-mass arithmetic."""

from __future__ import annotations

import pytest

from pharmaml.exceptions import UnitConversionError
from pharmaml.pharma.units import UnitConverter, molar_mass


class TestUnitConverter:
    def test_mass_scaling(self) -> None:
        assert UnitConverter.convert(500, "mg", "g") == pytest.approx(0.5)
        assert UnitConverter.convert(1, "kg", "mg") == pytest.approx(1e6)
        assert UnitConverter.convert(2, "µg", "ng") == pytest.approx(2000)

    def test_concentration_scaling(self) -> None:
        assert UnitConverter.convert(1.0, "uM", "nM") == pytest.approx(1000.0)
        assert UnitConverter.convert(250, "nM", "uM") == pytest.approx(0.25)

    def test_time_scaling(self) -> None:
        assert UnitConverter.convert(2, "h", "s") == pytest.approx(7200.0)
        assert UnitConverter.convert(90, "min", "h") == pytest.approx(1.5)

    def test_identity_conversion(self) -> None:
        assert UnitConverter.convert(3.14, "mg", "mg") == pytest.approx(3.14)

    def test_dimension_mismatch_rejected(self) -> None:
        with pytest.raises(UnitConversionError, match="cannot convert"):
            UnitConverter.convert(1.0, "mg", "mL")

    def test_unknown_unit_rejected(self) -> None:
        with pytest.raises(UnitConversionError, match="unknown unit"):
            UnitConverter.convert(1.0, "furlongs", "m")

    def test_dimension_lookup(self) -> None:
        assert UnitConverter.dimension_of("mL") == "volume"
        assert UnitConverter.dimension_of("mmol") == "amount"


class TestMolarMass:
    def test_water(self) -> None:
        assert molar_mass("H2O") == pytest.approx(18.015, rel=1e-4)

    def test_caffeine(self) -> None:
        assert molar_mass("C8H10N4O2") == pytest.approx(194.19, rel=1e-3)

    def test_parenthesised_group(self) -> None:
        # Ca(OH)2 = 40.078 + 2*(15.999 + 1.008) = 74.092
        assert molar_mass("Ca(OH)2") == pytest.approx(74.092, rel=1e-4)

    def test_hydrate_notation(self) -> None:
        # MgSO4.7H2O = 120.367 + 7*18.015 = 246.47
        assert molar_mass("MgSO4.7H2O") == pytest.approx(246.47, rel=1e-3)

    def test_single_atom(self) -> None:
        assert molar_mass("Na") == pytest.approx(22.990, rel=1e-4)

    def test_unknown_element(self) -> None:
        with pytest.raises(UnitConversionError, match="unknown element"):
            molar_mass("Xx2Y3")

    def test_malformed_formula(self) -> None:
        with pytest.raises(UnitConversionError):
            molar_mass("C8H10N4O2!!!")

    def test_unbalanced_parentheses(self) -> None:
        with pytest.raises(UnitConversionError, match="unbalanced"):
            molar_mass("Ca(OH2")

    def test_empty_formula(self) -> None:
        with pytest.raises(UnitConversionError):
            molar_mass("")


def test_conversion_within_dimension_is_transitive() -> None:
    step1 = UnitConverter.convert(7.5, "mg", "g")
    step2 = UnitConverter.convert(step1, "g", "kg")
    assert step2 == pytest.approx(UnitConverter.convert(7.5, "mg", "kg"))

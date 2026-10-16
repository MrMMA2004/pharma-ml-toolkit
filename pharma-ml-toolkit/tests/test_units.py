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


class TestMolarMassRobustness:
    """Malformed formulas must raise; documented examples must parse."""

    def test_docstring_example_cobalt_hexammine(self) -> None:
        # Regression: the _parse_formula_group docstring promises
        # Co(NH3)6.3H2O parses -- Co was missing from the element table.
        assert molar_mass("Co(NH3)6.3H2O") == pytest.approx(215.164, rel=1e-4)

    def test_empty_parenthesis_group_rejected(self) -> None:
        # Regression: "Ca()2" used to return the plain Ca mass silently.
        with pytest.raises(UnitConversionError, match="empty"):
            molar_mass("Ca()2")

    def test_dangling_hydrate_count_rejected(self) -> None:
        # Regression: "MgSO4.7" used to return the plain MgSO4 mass silently.
        with pytest.raises(UnitConversionError, match="hydrate"):
            molar_mass("MgSO4.7")

    def test_non_string_formula_rejected(self) -> None:
        # Regression: molar_mass(None) escaped as a raw AttributeError.
        with pytest.raises(UnitConversionError, match="string"):
            molar_mass(None)
        with pytest.raises(UnitConversionError, match="string"):
            molar_mass(194)

    def test_nested_parentheses(self) -> None:
        # PINNING (already correct before this round): K4(Fe(CN)6)
        # = 4 K + Fe + 6 (C + N).
        assert molar_mass("K4(Fe(CN)6)") == pytest.approx(
            4 * 39.098 + 55.845 + 6 * (12.011 + 14.007), rel=1e-4
        )


class TestAtomicWeightTable:
    """The element table must never drift from published values again.

    Regression (100x-22-c): the 19-a/20-a triage list flagged missing Cu
    and Ni; the full-table audit also found that platinum (cisplatin!),
    arsenic (arsenic trioxide), gadolinium (MRI contrast) and 14 other
    pharma-relevant elements were absent, so their formulas raised
    ``"unknown element"`` instead of computing a molar mass.
    """

    def test_every_entry_matches_published_weights(self) -> None:
        # Pin every entry against RDKit's periodic table (tolerance 0.01
        # covers rounding-convention differences).  Two INTENTIONAL
        # divergences: our Zn (65.38) and Se (78.971) follow the IUPAC
        # 2021 abridged values, which are NEWER than RDKit's 65.39 /
        # 78.96 -- documented, not drift.
        from rdkit import Chem

        from pharmaml.pharma.units import _ATOMIC_WEIGHTS

        pt = Chem.GetPeriodicTable()
        intentional = {"Zn", "Se"}
        for symbol, ours in _ATOMIC_WEIGHTS.items():
            z = pt.GetAtomicNumber(symbol)
            rdkit_w = pt.GetAtomicWeight(z)
            if symbol in intentional:
                # 0.012 absorbs the binary dust at the 0.011 Se delta
                assert abs(ours - rdkit_w) <= 0.012, (symbol, ours, rdkit_w)
            else:
                assert abs(ours - rdkit_w) <= 0.01, (symbol, ours, rdkit_w)

    def test_pharma_relevant_metals_present(self) -> None:
        from pharmaml.pharma.units import _ATOMIC_WEIGHTS

        required = {
            "Cu": 63.546, "Ni": 58.693, "Pt": 195.08, "Au": 196.967,
            "Ag": 107.868, "As": 74.922, "Bi": 208.980, "Gd": 157.25,
            "Ba": 137.327, "Ti": 47.867,
        }
        for symbol, weight in required.items():
            assert symbol in _ATOMIC_WEIGHTS, symbol
            assert _ATOMIC_WEIGHTS[symbol] == pytest.approx(weight, abs=1e-6)

    def test_real_metal_drug_formulas_compute(self) -> None:
        # cisplatin Pt(NH3)2Cl2 ~ 300.05 g/mol (published); arsenic
        # trioxide As2O3 (Trisenox) ~ 197.84; copper sulfate ~ 159.6;
        # barium sulfate radiocontrast ~ 233.4.
        assert molar_mass("Pt(NH3)2Cl2") == pytest.approx(300.05, abs=0.02)
        assert molar_mass("As2O3") == pytest.approx(197.84, abs=0.02)
        assert molar_mass("CuSO4") == pytest.approx(159.61, abs=0.02)
        assert molar_mass("BaSO4") == pytest.approx(233.39, abs=0.02)

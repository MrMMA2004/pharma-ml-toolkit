"""Unit conversions and molar-mass arithmetic.

A deliberate design choice: conversions go through a single
:func:`UnitConverter.convert` front-door with dimension checking,
instead of ad-hoc multiplication factors scattered through formulas
(a classic source of 1000-fold medication errors).
"""

from __future__ import annotations

import re
from functools import lru_cache

from pharmaml.exceptions import UnitConversionError

__all__ = ["UnitConverter", "molar_mass"]

# Canonical scales relative to the SI base unit of each dimension.
_SCALES: dict[str, dict[str, float]] = {
    "mass": {
        "kg": 1e3, "g": 1.0, "mg": 1e-3, "ug": 1e-6, "µg": 1e-6,
        "mcg": 1e-6, "ng": 1e-9, "pg": 1e-12,
    },
    "volume": {"L": 1.0, "mL": 1e-3, "ml": 1e-3, "uL": 1e-6, "µL": 1e-6, "nL": 1e-9},
    "amount": {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "µmol": 1e-6, "nmol": 1e-9},
    "concentration": {
        "M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6,
        "nM": 1e-9, "pM": 1e-12,
    },
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "d": 86400.0, "day": 86400.0},
}


class UnitConverter:
    """Dimension-aware unit converter for pharmaceutical quantities.

    Examples
    --------
    >>> UnitConverter.convert(500, "mg", "g")
    0.5
    >>> UnitConverter.convert(1.0, "uM", "nM")
    1000.0
    >>> UnitConverter.convert(1.0, "mg", "mL")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.UnitConversionError: cannot convert mass 'mg' to volume 'mL'
    """

    @classmethod
    def dimension_of(cls, unit: str) -> str:
        """Return the physical dimension a unit belongs to."""
        for dimension, scale in _SCALES.items():
            if unit in scale:
                return dimension
        supported = {
            dimension: sorted(set(units)) for dimension, units in _SCALES.items()
        }
        raise UnitConversionError(
            f"unknown unit {unit!r}; supported units by dimension: {supported}"
        )

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert ``value`` between units of the same dimension.

        Raises
        ------
        UnitConversionError
            If either unit is unknown or the dimensions differ.
        """
        source, target = from_unit.strip(), to_unit.strip()
        source_dim = cls.dimension_of(source)
        target_dim = cls.dimension_of(target)
        if source_dim != target_dim:
            raise UnitConversionError(
                f"cannot convert {source_dim} '{source}' to {target_dim} '{target}'"
            )
        return float(value) * _SCALES[source_dim][source] / _SCALES[target_dim][target]


# Chemical formula parser: e.g. "C8H10N4O2" -> {C:8, H:10, N:4, O:2}.
# Counts are integers; fractional notation and hydrate dots are handled
# by the group parser, not by this token pattern.
_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# Standard atomic weights (IUPAC 2021 abridged values).
_ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Co": 58.933, "Fe": 55.845, "Zn": 65.38, "Se": 78.971, "Br": 79.904,
    "I": 126.90,
}


@lru_cache(maxsize=1024)
def molar_mass(formula: str) -> float:
    """Compute the molar mass (g/mol) of a chemical formula.

    Supports nested hydrate notation such as ``MgSO4.7H2O`` and
    parentheses groups like ``Ca(OH)2``.

    Raises
    ------
    UnitConversionError
        If the formula is not a string, contains unknown elements or
        malformed groups (including empty ``()`` groups and hydrate
        counts not followed by a formula).

    Examples
    --------
    >>> round(molar_mass("H2O"), 3)
    18.015
    >>> round(molar_mass("C8H10N4O2"), 2)   # caffeine
    194.19
    """
    if not isinstance(formula, str):
        raise UnitConversionError(
            f"formula must be a string, got {type(formula).__name__}: {formula!r}"
        )
    text = formula.replace(" ", "").replace("·", ".")
    if not text:
        raise UnitConversionError("empty formula")
    return _parse_formula_group(text)


def _parse_formula_group(text: str) -> float:
    """Evaluate one (sub)formula string left-to-right.

    Handles parenthesis groups appearing anywhere (``Ca(OH)2``) and
    hydrate dot notation (``MgSO4.7H2O``) without pre-splitting, so
    mixed forms like ``Co(NH3)6.3H2O`` also parse correctly.
    """
    if not text:
        return 0.0

    total = 0.0
    pos = 0
    while pos < len(text):
        char = text[pos]

        if char == "(":
            # Find the matching closing parenthesis.
            depth = 0
            end = -1
            for scan in range(pos, len(text)):
                if text[scan] == "(":
                    depth += 1
                elif text[scan] == ")":
                    depth -= 1
                    if depth == 0:
                        end = scan
                        break
            if end == -1:
                raise UnitConversionError(f"unbalanced parentheses in formula {text!r}")
            inner = text[pos + 1 : end]
            if not inner:
                raise UnitConversionError(
                    f"empty parenthesis group in formula {text!r}"
                )
            multiplier_match = re.match(r"(\d+)", text[end + 1 :])
            multiplier_txt = multiplier_match.group(1) if multiplier_match else ""
            multiplier = float(multiplier_txt) if multiplier_txt else 1.0
            total += _parse_formula_group(inner) * multiplier
            pos = end + 1 + len(multiplier_txt)
            continue

        if char == ")":
            raise UnitConversionError(f"unbalanced parentheses in formula {text!r}")

        if char == ".":
            # Hydrate notation: ".7H2O" -> 7 x H2O.
            multiplier_match = re.match(r"\.(\d+)", text[pos:])
            if not multiplier_match:
                raise UnitConversionError(f"malformed hydrate notation in {text!r}")
            count = float(multiplier_match.group(1))
            tail = text[pos + multiplier_match.end() :]
            if not tail:
                raise UnitConversionError(
                    f"hydrate count {multiplier_match.group(1)} in {text!r} "
                    "is not followed by a formula"
                )
            total += count * _parse_formula_group(tail)
            break

        match = _ELEMENT_TOKEN.match(text, pos)
        if not match:
            raise UnitConversionError(f"malformed formula segment {text!r}")
        element, count_txt = match.group(1), match.group(2)
        if element not in _ATOMIC_WEIGHTS:
            raise UnitConversionError(
                f"unknown element {element!r} in formula {text[:30]!r}"
            )
        count = float(count_txt) if count_txt else 1.0
        total += _ATOMIC_WEIGHTS[element] * count
        pos = match.end()

    return total

"""100x-123 — unit-converter parity guard, Python side.

Emits /tmp/units_ref_123.json with the SAME matrix the TS twin emits:
every same-dimension pair of every dimension at the sample value 7.5,
the dimension_of answers, the sorted units-per-dimension lists, and
the two error paths' messages. The suite's TestUnitConverterParity
compares the twins field-by-field.

Run from the toolkit root: python scratch/probe_units_parity_123.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharmaml.exceptions import UnitConversionError  # noqa: E402
from pharmaml.pharma.units import _SCALES, UnitConverter  # noqa: E402

SAMPLE = 7.5

conversions = []
for dimension, scale in _SCALES.items():
    for source in scale:
        for target in scale:
            conversions.append(
                {
                    "from": source,
                    "to": target,
                    "value": UnitConverter.convert(SAMPLE, source, target),
                }
            )

dimensions = {}
for unit in ("kg", "mg", "\u00b5g", "mL", "uL", "mmol", "nM", "pM", "h", "day"):
    dimensions[unit] = UnitConverter.dimension_of(unit)

unit_lists = {dimension: sorted(scale) for dimension, scale in _SCALES.items()}


def error_of(fn) -> str:  # noqa: ANN001, ANN202
    try:
        fn()
        return "NO_ERROR"
    except UnitConversionError as exc:
        return str(exc)


errors = {
    "crossDimension": error_of(lambda: UnitConverter.convert(1, "mg", "mL")),
    "unknownUnit": error_of(lambda: UnitConverter.dimension_of("bogus")),
}

Path("/tmp/units_ref_123.json").write_text(
    json.dumps(
        {
            "conversions": conversions,
            "dimensions": dimensions,
            "unitLists": unit_lists,
            "errors": errors,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(
    f"units parity (python): {len(conversions)} conversions, "
    f"{len(dimensions)} dimension_of answers, 2 error paths"
)

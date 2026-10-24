/**
 * 100x-123 — unit-converter parity guard, TS side.
 *
 * Imports the SHIPPED src/lib/unit-converter.ts (the module the
 * Calculators card imports, not a mirror) and emits the same JSON
 * matrix the Python twin emits (/tmp/units_ts_123.json): every
 * same-dimension pair of every dimension at a sample value, the
 * dimension_of answers, the units-per-dimension lists, and the two
 * error paths' messages. The suite's TestUnitConverterParity compares
 * the twins field-by-field — a factor drift in either language fails
 * CI with the diverging row named.
 *
 * Run: bun run scratch/probe_units_parity_123.ts
 */
import { writeFileSync } from "fs";
import {
  UNIT_SCALES,
  UnitConversionError,
  convertUnits,
  dimensionOf,
  unitsForDimension,
  type Dimension,
} from "../src/lib/unit-converter";

const SAMPLE = 7.5;
const conversions: { from: string; to: string; value: number }[] = [];
for (const dimension of Object.keys(UNIT_SCALES) as Dimension[]) {
  const units = Object.keys(UNIT_SCALES[dimension]);
  for (const from of units) {
    for (const to of units) {
      conversions.push({ from, to, value: convertUnits(SAMPLE, from, to) });
    }
  }
}

const dimensions: Record<string, string> = {};
for (const unit of ["kg", "mg", "µg", "mL", "uL", "mmol", "nM", "pM", "h", "day"]) {
  dimensions[unit] = dimensionOf(unit);
}

const unitLists: Record<string, string[]> = {};
for (const dimension of Object.keys(UNIT_SCALES) as Dimension[]) {
  unitLists[dimension] = unitsForDimension(dimension);
}

function errorOf(fn: () => void): string {
  try {
    fn();
    return "NO_ERROR";
  } catch (error) {
    if (error instanceof UnitConversionError) return error.message;
    return `WRONG_ERROR_TYPE: ${String(error)}`;
  }
}

const errors = {
  crossDimension: errorOf(() => convertUnits(1, "mg", "mL")),
  unknownUnit: errorOf(() => dimensionOf("bogus")),
};

writeFileSync(
  "/tmp/units_ts_123.json",
  JSON.stringify({ conversions, dimensions, unitLists, errors }, null, 2),
);
console.log(
  `units parity (ts): ${conversions.length} conversions, ${Object.keys(dimensions).length} dimension_of answers, 2 error paths`,
);

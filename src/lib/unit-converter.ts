/**
 * Unit conversions — the browser twin of `pharmaml.pharma.units`
 * (the 100x-123 pair).
 *
 * The Python module's design choice is carried over: conversions go
 * through one `convertUnits` front door with dimension checking rather
 * than ad-hoc factors. `TestUnitConverterParity` pins exact conversion
 * values and the listed dimension/error cases: every same-dimension
 * pair at 7.5, selected direct dimension lookups, sorted unit lists,
 * and cross-/unknown-unit errors. This is a bounded probe, not a claim
 * that every call behaves identically: both conversion front doors trim
 * their unit arguments, but this exported `dimensionOf` trims a direct
 * argument while Python's `UnitConverter.dimension_of` does not.
 * `molarMass` stays Python-side (the calculators tab has no formula
 * input — it is one API route away if a card ever needs it).
 */

export type Dimension = "mass" | "volume" | "amount" | "concentration" | "time";

/** Canonical scales relative to the SI base unit of each dimension. */
export const UNIT_SCALES: Record<Dimension, Record<string, number>> = {
  mass: {
    kg: 1e3, g: 1.0, mg: 1e-3, ug: 1e-6, "µg": 1e-6,
    mcg: 1e-6, ng: 1e-9, pg: 1e-12,
  },
  volume: { L: 1.0, mL: 1e-3, ml: 1e-3, uL: 1e-6, "µL": 1e-6, nL: 1e-9 },
  amount: { mol: 1.0, mmol: 1e-3, umol: 1e-6, "µmol": 1e-6, nmol: 1e-9 },
  concentration: {
    M: 1.0, mM: 1e-3, uM: 1e-6, "µM": 1e-6,
    nM: 1e-9, pM: 1e-12,
  },
  time: { s: 1.0, min: 60.0, h: 3600.0, hr: 3600.0, d: 86400.0, day: 86400.0 },
};

/** Mirrors pharmaml.exceptions.UnitConversionError (message parity). */
export class UnitConversionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnitConversionError";
  }
}

/** The physical dimension a unit belongs to (throws on unknown units). */
export function dimensionOf(unit: string): Dimension {
  const trimmed = unit.trim();
  for (const [dimension, scale] of Object.entries(UNIT_SCALES) as [
    Dimension,
    Record<string, number>,
  ][]) {
    if (trimmed in scale) return dimension;
  }
  const supported: Record<Dimension, string[]> = {
    mass: unitsForDimension("mass"),
    volume: unitsForDimension("volume"),
    amount: unitsForDimension("amount"),
    concentration: unitsForDimension("concentration"),
    time: unitsForDimension("time"),
  };
  throw new UnitConversionError(
    `unknown unit '${trimmed}'; supported units by dimension: ${JSON.stringify(supported)}`,
  );
}

/** Sorted unit spellings of one dimension (the hint line's source). */
export function unitsForDimension(dimension: Dimension): string[] {
  return Object.keys(UNIT_SCALES[dimension]).sort();
}

/** Convert `value` between units of the same dimension.
 *
 * @throws UnitConversionError if either unit is unknown or the
 *   dimensions differ — with the library's exact message text.
 */
export function convertUnits(value: number, fromUnit: string, toUnit: string): number {
  const source = fromUnit.trim();
  const target = toUnit.trim();
  const sourceDim = dimensionOf(source);
  const targetDim = dimensionOf(target);
  if (sourceDim !== targetDim) {
    throw new UnitConversionError(
      `cannot convert ${sourceDim} '${source}' to ${targetDim} '${target}'`,
    );
  }
  return (value * UNIT_SCALES[sourceDim][source]) / UNIT_SCALES[targetDim][target];
}

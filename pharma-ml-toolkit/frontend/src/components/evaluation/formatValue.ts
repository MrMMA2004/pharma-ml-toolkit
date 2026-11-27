/**
 * One table cell's honest rendering (§17.3: type-aware, bounded).
 * Extracted from the ML workspace (part21 §3.1's shared layer) so
 * the evaluation inspector and the pages share one implementation.
 */
export function formatEvalValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return typeof value === "number"
    ? Number.isInteger(value)
      ? String(value)
      : value.toFixed(4)
    : String(value);
}

/**
 * 100x-153 — Pearson correlation parity, TS side.
 *
 * Imports the SHIPPED src/lib/ml-math.ts pearsonCorrelation (added this
 * round for the ML playground's Feature-target correlations card — that
 * is the whole point: the probe runs the code the browser runs, not a
 * mirror) and checks it against the scipy.stats.pearsonr gold written
 * by the Python twin
 * (pharma-ml-toolkit/scratch/verify_corr_parity_153.py).
 *
 * Tolerance: |r_ts − r_scipy| ≤ 1e-12 on every defined pair (pure
 * two-pass arithmetic). The undefined cases — a length mismatch (scipy
 * raises; the TS must NOT silently truncate to the common prefix), a
 * constant column, fewer than two points, non-finite inputs — must be
 * NaN on both sides, never a fake 0.
 *
 * Run AFTER the python twin:
 *   bun run scratch/verify_corr_parity_153.ts
 */
import { pearsonCorrelation } from "../src/lib/ml-math";
import { readFileSync, writeFileSync } from "node:fs";

interface Case {
  name: string;
  a: (number | string)[];
  b: (number | string)[];
}
interface RefEntry {
  name: string;
  r: number | null;
}

/** Decode the py twin's NaN/Infinity string sentinels back to numbers. */
function decodeColumn(values: (number | string)[]): number[] {
  return values.map((v) =>
    typeof v === "number"
      ? v
      : v === "NaN"
        ? Number.NaN
        : v === "Infinity"
          ? Number.POSITIVE_INFINITY
          : Number.NEGATIVE_INFINITY,
  );
}

const cases = JSON.parse(readFileSync("/tmp/corr_inputs_153.json", "utf8")) as Case[];
const ref = JSON.parse(readFileSync("/tmp/corr_ref_153.json", "utf8")) as RefEntry[];

const TOL = 1e-12;
const out: RefEntry[] = [];
let worst = 0;
for (const c of cases) {
  const r = pearsonCorrelation(decodeColumn(c.a), decodeColumn(c.b));
  out.push({ name: c.name, r: Number.isFinite(r) ? r : null });
}
writeFileSync("/tmp/corr_ts_153.json", JSON.stringify(out));

for (const [entry, refEntry] of out.map((e, i) => [e, ref[i]] as const)) {
  if (entry.r === null || refEntry.r === null) {
    if (!(entry.r === null && refEntry.r === null)) {
      throw new Error(
        `${refEntry.name}: undefined mismatch ts=${entry.r} scipy=${refEntry.r} — the TS must be NaN exactly where scipy is`,
      );
    }
    continue;
  }
  const d = Math.abs(entry.r - refEntry.r);
  if (d > worst) worst = d;
}
console.log(`ts-vs-scipy worst |d| = ${worst.toExponential(3)} over ${out.length} case(s)`);
if (worst > TOL) {
  throw new Error(`the SHIPPED TS drifted from scipy: ${worst.toExponential(3)} > ${TOL.toExponential(1)}`);
}
console.log("TS_PARITY_OK");

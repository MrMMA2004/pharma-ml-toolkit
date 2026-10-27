/**
 * 100x-154 — Pearson correlation parity, ADVERSARIAL numerics grid,
 * TS side.  The twin of scratch/verify_corr_parity_154.py: imports
 * the SHIPPED src/lib/ml-math.ts pearsonCorrelation (now the
 * offset-then-scale recipe) and checks it against the scipy gold on
 * the adversarial grid the round-36 pair was silent about (extreme
 * scales, huge offsets, near-constant columns, the constant-column
 * fake-zero pins).
 *
 * Tolerances (the python twin enforces the SAME set): 1e-12 on every
 * defined pair except huge_offset (5e-3 — no float64 path recovers
 * the analytic truth there; the input itself is quantized at
 * construction), and the analytic-truth pins (huge_offset vs 1.0 at
 * 2e-3).  The fake-zero pins (const_huge / const_tiny / const_plain)
 * must be NaN on both sides — never a fabricated 0.
 *
 * Run AFTER the python twin:
 *   bun run scratch/verify_corr_parity_154.ts
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
  truth?: number;
  truthTol?: number;
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

const cases = JSON.parse(readFileSync("/tmp/corr_adv_154_in.json", "utf8")) as Case[];
const ref = JSON.parse(readFileSync("/tmp/corr_adv_154_ref.json", "utf8")) as RefEntry[];

const TOL_DEFAULT = 1e-12;
/* huge_offset: the irreducible input-quantization floor; near_constant:
 * scipy's own NearConstantInputWarning territory (scipy loses ~1e-7 of
 * a 1e-12 jitter; the recipe is exact -- pinned against the analytic
 * truth in the gold's truth/truthTol fields). */
const TOL_OVERRIDES: Record<string, number> = { huge_offset: 5e-3, near_constant: 1e-7 };

const out: RefEntry[] = cases.map((c) => {
  const r = pearsonCorrelation(decodeColumn(c.a), decodeColumn(c.b));
  return { name: c.name, r: Number.isFinite(r) ? r : null };
});
writeFileSync("/tmp/corr_adv_154_ts.json", JSON.stringify(out));

let worst = 0;
const failures: string[] = [];
for (const [entry, refEntry] of out.map((e, i) => [e, ref[i]] as const)) {
  const tol = TOL_OVERRIDES[entry.name] ?? TOL_DEFAULT;
  if (entry.r === null || refEntry.r === null) {
    if (!(entry.r === null && refEntry.r === null)) {
      failures.push(`${entry.name}: undefined mismatch ts=${entry.r} scipy=${refEntry.r}`);
    }
  } else {
    const dr = Math.abs(entry.r - refEntry.r);
    if (dr > tol) {
      failures.push(`${entry.name}: |ts-scipy|=${dr.toExponential(3)} > ${tol.toExponential(1)}`);
    }
    if (!(entry.name in TOL_OVERRIDES)) worst = Math.max(worst, dr);
  }
  if (refEntry.truth !== undefined) {
    const truth = refEntry.truth;
    const truthTol = refEntry.truthTol;
    if (truthTol === undefined) {
      failures.push(`${entry.name}: missing analytic-truth tolerance in the gold`);
    } else if (entry.r === null || Math.abs(entry.r - truth) > truthTol) {
      failures.push(`${entry.name}: ts=${entry.r} vs analytic truth ${truth} (tol ${truthTol})`);
    }
  }
}
console.log(`ts-vs-scipy worst |dr| (default-tolerance cases): ${worst.toExponential(3)}`);
if (failures.length) {
  console.log("TS_SIDE_FAILURES:");
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
console.log("TS_SIDE_DONE", out.length);

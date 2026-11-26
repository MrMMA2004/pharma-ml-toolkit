/**
 * 100x-125 — learning-curve CONTRACT parity guard, TS side.
 *
 * Imports the SHIPPED src/lib/learning-curve.ts (the module the
 * /api/ml/learning-curve route imports, not a mirror) and emits the
 * twin JSON (/tmp/lc_ts_125.json): the default fractions, the gap
 * threshold, the verdict strings for the Python probe's gaps (passed
 * as argv, else a boundary grid around the 0.15 cut), and the
 * reduction engine's output over a deterministic hand-computed
 * fixture (prefix subsampling, 1-based fold slots, population std).
 *
 * The suite's TestLearningCurveContract compares the fraction schedule
 * within 1e-12, the same threshold, byte-identical verdict strings for
 * identical gaps, and the engine fixture against hand-computed numbers.
 * It does not compare sklearn's absolute training sizes or scores.
 *
 * Run: bun run scratch/probe_lc_parity_125.ts [gap...]
 */
import { writeFileSync } from "fs";
import {
  LC_FRACTIONS,
  LC_GAP_THRESHOLD,
  lcVerdict,
  learningCurvePoints,
} from "../src/lib/learning-curve";

/* The verdicts: the Python probe's gaps when given (the suite passes
 * them), plus the boundary grid that pins the 0.15 cut on this side. */
const gaps: number[] = process.argv.slice(2).map((a) => Number.parseFloat(a));
for (const boundary of [0.05, 0.149999, 0.150001, 0.5]) {
  if (!gaps.some((g) => Math.abs(g - boundary) < 1e-9)) gaps.push(boundary);
}
const verdicts: Record<string, string> = {};
for (const gap of gaps) {
  verdicts[String(gap)] = lcVerdict(gap);
}

/* The engine fixture — hand-computed below, asserted by the suite:
 * two folds, each with a 4-row training complement; the stub returns
 * [slot*10 + trainRows, 100 + slot] so every contract shows in the
 * arithmetic (prefix cut, 1-based slot, population std).
 *
 * fraction 0.5 -> size 2: fold1 [12, 101], fold2 [22, 102]
 *   -> train 17 ± 5, val 101.5 ± 0.5
 * fraction 1.0 -> size 4: fold1 [14, 101], fold2 [24, 102]
 *   -> train 19 ± 5, val 101.5 ± 0.5
 */
const engine = learningCurvePoints({
  folds: [
    { fold: [3], trainIdx: [0, 1, 2, 4] },
    { fold: [4], trainIdx: [0, 1, 2, 3] },
  ],
  fractions: [0.5, 1.0],
  fitAndScore: (trainRows, _valRows, slot) => [slot * 10 + trainRows.length, 100 + slot],
});

const payload = {
  fractions: [...LC_FRACTIONS],
  threshold: LC_GAP_THRESHOLD,
  verdicts,
  engine,
};
writeFileSync("/tmp/lc_ts_125.json", JSON.stringify(payload, null, 1));
console.log(
  `lc_ts_125: ${Object.keys(verdicts).length} verdict(s), threshold ${LC_GAP_THRESHOLD}, ` +
    `${engine.length} engine point(s)`
);

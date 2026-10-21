/**
 * 100x-114 — EI/UCB acquisition-math parity, TS side.
 *
 * Imports the SHIPPED src/lib/bayes-acquisition.ts (extracted from
 * ml-playground.tsx this round — that is the whole point: the probe
 * runs the code the browser runs, not a mirror) and checks it against
 * the scipy.stats.norm gold values written by the Python twin
 * (pharma-ml-toolkit/scratch/verify_acquisition_parity_114.py).
 *
 * Tolerances: Φ ≤ 1.5e-7 (AS-26.2.17 bound 7.5e-8, ×2 margin),
 * φ ≤ 1e-12, EI ≤ 1e-6, UCB ≤ 1e-12 (pure arithmetic).
 *
 * Run AFTER the python twin:
 *   bun run scratch/verify_acquisition_parity_114.ts
 */
import {
  expectedImprovement,
  stdNormalCdf,
  stdNormalPdf,
  upperConfidenceBound,
} from "../src/lib/bayes-acquisition";

import { readFileSync, writeFileSync } from "node:fs";

interface Grid {
  z: number[];
  ei: number[][]; // [mu, sigma, fStar]
  ucb: number[][]; // [mu, sigma, kappa]
}
interface Ref {
  cdf: number[];
  pdf: number[];
  ei: number[];
  ucb: number[];
}

const grid = JSON.parse(readFileSync("/tmp/acq_inputs_114.json", "utf8")) as Grid;
const ref = JSON.parse(readFileSync("/tmp/acq_ref_114.json", "utf8")) as Ref;

const cdf = grid.z.map(stdNormalCdf);
const pdf = grid.z.map(stdNormalPdf);
const ei = grid.ei.map(([mu, sigma, fStar]) => expectedImprovement(mu, sigma, fStar));
const ucb = grid.ucb.map(([mu, sigma, kappa]) => upperConfidenceBound(mu, sigma, kappa));

writeFileSync("/tmp/acq_ts_114.json", JSON.stringify({ cdf, pdf, ei, ucb }));

const maxAbs = (a: number[], b: number[]) =>
  a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

const tols: Record<string, number> = { cdf: 1.5e-7, pdf: 1e-12, ei: 1e-6, ucb: 1e-12 };
const diffs = {
  cdf: maxAbs(cdf, ref.cdf),
  pdf: maxAbs(pdf, ref.pdf),
  ei: maxAbs(ei, ref.ei),
  ucb: maxAbs(ucb, ref.ucb),
};

let failed = false;
for (const k of ["cdf", "pdf", "ei", "ucb"] as const) {
  const ok = diffs[k] <= tols[k];
  if (!ok) failed = true;
  console.log(
    `  ${k.padEnd(3)}: max|d|=${diffs[k].toExponential(3)} tol=${tols[k].toExponential(1)} ${ok ? "OK" : "FAIL"}`,
  );
}

// guard-branch spot check: sigma under the 1e-12 threshold collapses
// to the deterministic improvement (rows 73..75 of the grid are the
// degenerate cases appended by the python twin)
const guard = [2.0, 1e-12, 1.0];
const guardVal = expectedImprovement(guard[0], guard[1], guard[2]);
const guardOk = Math.abs(guardVal - 1.0) < 1e-15; // max(0, 2-1) = 1
console.log(
  `  guard: EI(2, 1e-12, 1) = ${guardVal} (expect 1) ${guardOk ? "OK" : "FAIL"}`,
);
if (!guardOk) failed = true;

if (failed) {
  console.error("ACQUISITION PARITY: FAIL");
  process.exit(1);
}
console.log("ACQUISITION PARITY: PASS (shipped TS module vs scipy, all four quantities)");

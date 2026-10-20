/* 105 verification: the web GP-classifier fitter vs sklearn's
 * GaussianProcessClassifier (Laplace).
 *
 * Fits fitGaussianProcessClassification on a fixed binary dataset with
 * FIXED kernel hyperparameters (no grid search — kernelTuned must be
 * false so both implementations solve the same algebra), predicts
 * probabilities on a fixed grid, and dumps them as JSON for the Python
 * side (scratch/verify_gpc_parity.py --check-ts) to compare against
 * sklearn. The only approximation between the two is erf itself
 * (A&S 7.1.26, |ε| ≤ 1.5e-7 → proba parity ≲ 5e-7).
 *
 * Run: bun run scratch/verify_gpc_parity.ts (from the web root)
 */

import { fitGaussianProcessClassification } from "../src/lib/ml-math";

/* Fixed dataset — 60 train rows, 2 features, deterministic values
 * (no RNG: hand-spread numbers so both sides see identical bytes). */
const X: number[][] = [];
const y: number[] = [];
for (let i = 0; i < 60; i++) {
  const a = -2 + (((i * 37) % 41) / 41) * 4;
  const b = -1.5 + (((i * 23) % 53) / 53) * 3;
  X.push([a, b]);
  /* Two half-moons-ish labels: deterministic, overlapping. */
  const upper = b > 0.35 + 0.55 * Math.sin(a);
  y.push(upper ? 1 : 0);
}

const grid: number[][] = [];
for (let i = 0; i < 25; i++) {
  grid.push([-2 + (((i * 29) % 37) / 37) * 4, -1.5 + (((i * 17) % 31) / 31) * 3]);
}

const fit = fitGaussianProcessClassification(X, y, {
  lengthScale: 0.8,
  signalStd: 1.2,
});

const probas = grid.map((g) => fit.predictProba(g));
const preds = grid.map((g) => fit.predict01(g));

const out = {
  kernelTuned: fit.kernelTuned,
  lengthScale: fit.lengthScale,
  signalStd: fit.signalStd,
  newtonIterations: fit.newtonIterations,
  logMarginalLikelihood: fit.logMarginalLikelihood,
  nTrain: X.length,
  probas,
  preds,
};

console.log(JSON.stringify(out, null, 2));

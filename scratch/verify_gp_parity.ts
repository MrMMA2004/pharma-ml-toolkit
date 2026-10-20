/* 102 verification: the web GP fitter vs sklearn's GaussianProcessRegressor.
 *
 * Fits fitGaussianProcess on a fixed 2-feature dataset with FIXED kernel
 * hyperparameters (no LML search — kernelTuned must be false so the two
 * implementations solve the SAME linear algebra), predicts on a fixed
 * grid, and dumps predictions + sigmas + LML as JSON for the Python side
 * to compare against sklearn to machine precision.
 *
 * Run: bun run scratch/verify_gp_parity.ts (from the web root)
 */

import { fitGaussianProcess } from "../src/lib/ml-math";

/* Fixed dataset — 40 train rows, 2 features, deterministic values
 * (no RNG: hand-spread numbers so both sides see identical bytes). */
const X: number[][] = [];
const y: number[] = [];
for (let i = 0; i < 40; i++) {
  const a = -2 + (i * 0.137) % 4;
  const b = ((i * 31) % 17) / 7 - 1;
  X.push([a, b]);
  y.push(1.7 * a - 0.9 * b + 0.4 + ((i * 13) % 11) / 55);
}

const grid: number[][] = [];
for (let i = 0; i < 25; i++) {
  grid.push([-2 + i * 0.17, ((i * 7) % 13) / 11 - 0.6]);
}

const fit = fitGaussianProcess(X, y, {
  lengthScale: 0.8,
  signalStd: 1.2,
  noiseStd: 0.25,
});

const out = {
  kernelTuned: fit.kernelTuned,
  lengthScale: fit.lengthScale,
  signalStd: fit.signalStd,
  noiseStd: fit.noiseStd,
  logMarginalLikelihood: fit.logMarginalLikelihood,
  nSupport: fit.nSupport,
  train: { X, y },
  grid,
  preds: grid.map((g) => fit.predict(g)),
  sigmas: grid.map((g) => fit.predictStd(g)),
};
console.log(JSON.stringify(out));

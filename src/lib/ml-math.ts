/**
 * Pure, deterministic math for the ML training playground API.
 * Everything here is side-effect free and driven by a seeded PRNG
 * (mulberry32) so the same request always yields the same dataset.
 */

/* ------------------------------------------------------------------ */
/* Random number generation                                            */
/* ------------------------------------------------------------------ */

/** Deterministic 32-bit PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draw via Box–Muller, consuming two uniforms. */
export function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Seeded Fisher–Yates shuffle of [0, n). */
export function shuffledIndices(n: number, rand: () => number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

/** Split indices into train/test sets using a seeded shuffle. */
export function shuffledSplit(
  n: number,
  testFraction: number,
  rand: () => number
): { train: number[]; test: number[] } {
  const idx = shuffledIndices(n, rand);
  const nTest = Math.min(Math.max(1, Math.round(n * testFraction)), n - 2);
  return { test: idx.slice(0, nTest), train: idx.slice(nTest) };
}

/** k contiguous folds over a seeded shuffle of [0, n). */
export function kFoldSplit(n: number, k: number, rand: () => number): number[][] {
  const idx = shuffledIndices(n, rand);
  const base = Math.floor(n / k);
  const rem = n % k;
  const folds: number[][] = [];
  let start = 0;
  for (let f = 0; f < k; f++) {
    const size = base + (f < rem ? 1 : 0);
    folds.push(idx.slice(start, start + size));
    start += size;
  }
  return folds;
}

/* ------------------------------------------------------------------ */
/* Linear algebra                                                      */
/* ------------------------------------------------------------------ */

/**
 * Solve A·x = b via Gaussian elimination with partial pivoting.
 * @throws Error when the system is (numerically) singular.
 */
export function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) {
      throw new Error("Singular normal-equation matrix — try different parameters.");
    }
    if (pivot !== col) {
      const tmp = M[col];
      M[col] = M[pivot];
      M[pivot] = tmp;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/* ------------------------------------------------------------------ */
/* Regression                                                          */
/* ------------------------------------------------------------------ */

export interface RegressionModel {
  mu: number;
  sigma: number;
  degree: number;
  /** w[0] = intercept on standardized features, then powers of z. */
  w: number[];
  predict(xRaw: number): number;
}

/** Mean and population standard deviation helper. */
function stats(values: number[]): { mu: number; sigma: number } {
  const n = values.length;
  const mu = values.reduce((a, b) => a + b, 0) / n;
  const sigma = Math.sqrt(values.reduce((a, b) => a + (b - mu) ** 2, 0) / n);
  return { mu, sigma: sigma > 1e-12 ? sigma : 1 };
}

/**
 * Fit least-squares (optionally ridge-regularized) polynomial regression.
 * x is standardized with train-set statistics; the intercept is never
 * penalized. Solved in closed form via normal equations.
 */
export function fitLinearRegression(
  xs: number[],
  ys: number[],
  opts: { alpha?: number; degree?: number } = {}
): RegressionModel {
  const alpha = opts.alpha ?? 0;
  const degree = opts.degree ?? 1;
  const n = xs.length;
  if (n < degree + 2) throw new Error("Not enough samples to fit the model.");
  const { mu, sigma } = stats(xs);

  const p = degree + 1;
  const A: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b: number[] = new Array<number>(p).fill(0);

  for (let i = 0; i < n; i++) {
    const z = (xs[i] - mu) / sigma;
    const row: number[] = [1];
    let zp = 1;
    for (let d = 1; d <= degree; d++) {
      zp *= z;
      row.push(zp);
    }
    for (let j = 0; j < p; j++) {
      b[j] += row[j] * ys[i];
      for (let k = j; k < p; k++) {
        const v = row[j] * row[k];
        A[j][k] += v;
        if (k !== j) A[k][j] += v;
      }
    }
  }
  for (let j = 1; j < p; j++) A[j][j] += alpha; // ridge: skip intercept

  const w = solveLinearSystem(A, b);
  return {
    mu,
    sigma,
    degree,
    w,
    predict(xRaw: number) {
      const z = (xRaw - mu) / sigma;
      let acc = w[0];
      let zp = 1;
      for (let d = 1; d <= degree; d++) {
        zp *= z;
        acc += w[d] * zp;
      }
      return acc;
    },
  };
}

export interface MultivariateModel {
  /** Per-feature train-set means (raw units). */
  mus: number[];
  /** Per-feature train-set population stds (raw units). */
  sigmas: number[];
  /** w[0] = intercept on standardized features, then one weight per feature. */
  w: number[];
  /** Predict from raw (unstandardized) feature values. */
  predict(xsRaw: number[]): number;
}

/**
 * Fit least-squares (optionally ridge-regularized) multivariate linear
 * regression on k features. Mirrors fitLinearRegression's conventions:
 * each feature is standardized with train-set statistics, the intercept is
 * never penalized, and the system is solved in closed form via normal
 * equations (Gaussian elimination with partial pivoting).
 */
export function fitMultivariateRegression(
  X: number[][],
  y: number[],
  opts: { alpha?: number } = {}
): MultivariateModel {
  const alpha = opts.alpha ?? 0;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  if (n < k + 2) {
    throw new Error(`Not enough samples to fit the model (${n} rows, ${k} features).`);
  }

  const mus: number[] = [];
  const sigmas: number[] = [];
  for (let j = 0; j < k; j++) {
    const col = X.map((row) => row[j] ?? 0);
    const s = stats(col);
    mus.push(s.mu);
    sigmas.push(s.sigma);
  }

  const p = k + 1;
  const A: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b: number[] = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    const row: number[] = [1];
    for (let j = 0; j < k; j++) {
      row.push(((X[i]?.[j] ?? 0) - mus[j]!) / sigmas[j]!);
    }
    for (let j = 0; j < p; j++) {
      b[j] += row[j]! * (y[i] ?? 0);
      for (let c = j; c < p; c++) {
        const v = row[j]! * row[c]!;
        A[j]![c] += v;
        if (c !== j) A[c]![j] += v;
      }
    }
  }
  for (let j = 1; j < p; j++) A[j]![j] += alpha; // ridge: skip intercept

  const w = solveLinearSystem(A, b);
  return {
    mus,
    sigmas,
    w,
    predict(xsRaw: number[]) {
      let acc = w[0] ?? 0;
      for (let j = 0; j < k; j++) {
        const z = ((xsRaw[j] ?? 0) - mus[j]!) / sigmas[j]!;
        acc += (w[j + 1] ?? 0) * z;
      }
      return acc;
    },
  };
}

export interface RegressionMetrics {
  rmse: number;
  mae: number;
  r2: number;
}

export function regressionMetrics(yTrue: number[], yPred: number[]): RegressionMetrics {
  const n = yTrue.length;
  let se = 0;
  let ae = 0;
  let sst = 0;
  const ym = yTrue.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    se += (yTrue[i] - yPred[i]) ** 2;
    ae += Math.abs(yTrue[i] - yPred[i]);
    sst += (yTrue[i] - ym) ** 2;
  }
  return {
    rmse: Math.sqrt(se / n),
    mae: ae / n,
    r2: sst > 1e-12 ? 1 - se / sst : 0,
  };
}

/** The noiseless generating line behind makeRegressionData: y = 0.8x + 0.5. */
export function regressionTrueY(x: number): number {
  return 0.8 * x + 0.5;
}

/** y = regressionTrueY(x) + noise·N(0,1), x ~ U(-3, 3). */
export function makeRegressionData(
  n: number,
  noise: number,
  rand: () => number
): { xs: number[]; ys: number[] } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = rand() * 6 - 3;
    const y = regressionTrueY(x) + noise * gaussian(rand);
    xs.push(x);
    ys.push(y);
  }
  return { xs, ys };
}

/* ------------------------------------------------------------------ */
/* Classification (two moons + logistic regression)                    */
/* ------------------------------------------------------------------ */

export interface MoonPoint {
  x: number;
  y: number;
  label: number;
}

/** Standard two-moons generator (sklearn-compatible geometry). */
export function makeMoons(n: number, noise: number, rand: () => number): MoonPoint[] {
  const nOut = Math.floor(n / 2);
  const nIn = n - nOut;
  const pts: MoonPoint[] = [];
  for (let i = 0; i < nOut; i++) {
    const t = (Math.PI * i) / Math.max(nOut - 1, 1);
    pts.push({ x: Math.cos(t), y: Math.sin(t), label: 0 });
  }
  for (let i = 0; i < nIn; i++) {
    const t = (Math.PI * i) / Math.max(nIn - 1, 1);
    pts.push({ x: 1 - Math.cos(t), y: 0.5 - Math.sin(t), label: 1 });
  }
  if (noise > 0) {
    for (const p of pts) {
      p.x += noise * gaussian(rand);
      p.y += noise * gaussian(rand);
    }
  }
  return pts;
}

export interface LogisticModel {
  mu: [number, number];
  sigma: [number, number];
  w: [number, number];
  bias: number;
  /** Predicted probability of class 1. */
  predict(xRaw: number, yRaw: number): number;
}

/**
 * Batch gradient-descent logistic regression with L2 regularization
 * (bias not penalized). Features are standardized with train statistics.
 */
export function fitLogisticRegression(
  pts: Array<{ x: number; y: number; label: number }>,
  opts: { l2?: number; lr?: number; epochs?: number } = {}
): LogisticModel {
  const l2 = opts.l2 ?? 1;
  const lr = opts.lr ?? 0.1;
  const epochs = opts.epochs ?? 800;
  const n = pts.length;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const sx = stats(xs);
  const sy = stats(ys);

  const F = pts.map((p) => [(p.x - sx.mu) / sx.sigma, (p.y - sy.mu) / sy.sigma]);
  const labels = pts.map((p) => p.label);

  let w0 = 0;
  let w1 = 0;
  let bias = 0;
  for (let e = 0; e < epochs; e++) {
    let gw0 = 0;
    let gw1 = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const z = bias + w0 * F[i][0] + w1 * F[i][1];
      const p = 1 / (1 + Math.exp(-z));
      const err = p - labels[i];
      gb += err;
      gw0 += err * F[i][0];
      gw1 += err * F[i][1];
    }
    bias -= (lr * gb) / n;
    w0 -= lr * (gw0 / n + l2 * w0);
    w1 -= lr * (gw1 / n + l2 * w1);
  }
  return {
    mu: [sx.mu, sy.mu],
    sigma: [sx.sigma, sy.sigma],
    w: [w0, w1],
    bias,
    predict(xRaw, yRaw) {
      const zx = (xRaw - sx.mu) / sx.sigma;
      const zy = (yRaw - sy.mu) / sy.sigma;
      const z = bias + w0 * zx + w1 * zy;
      return 1 / (1 + Math.exp(-z));
    },
  };
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export function classificationMetrics(yTrue: number[], yPred: number[]): ClassificationMetrics {
  const n = yTrue.length;
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < n; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else fn++;
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  return {
    accuracy: (tp + tn) / n,
    precision,
    recall,
    f1: precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0,
  };
}

/* ------------------------------------------------------------------ */
/* Feature-matrix standardization (train-set statistics)               */
/* ------------------------------------------------------------------ */

/** Column-wise train-set μ/σ used to map raw feature rows into the
 *  standardized space every multivariate trainer below works in. */
export interface Standardizer {
  mus: number[];
  sigmas: number[];
  /** Map one raw feature row to standardized space. */
  transform(xsRaw: number[]): number[];
}

export function fitStandardizer(X: number[][]): Standardizer {
  const k = X[0]?.length ?? 0;
  const mus: number[] = [];
  const sigmas: number[] = [];
  for (let j = 0; j < k; j++) {
    const s = stats(X.map((row) => row[j] ?? 0));
    mus.push(s.mu);
    sigmas.push(s.sigma);
  }
  return {
    mus,
    sigmas,
    transform(xsRaw: number[]) {
      const out: number[] = [];
      for (let j = 0; j < k; j++) {
        out.push(((xsRaw[j] ?? 0) - mus[j]!) / sigmas[j]!);
      }
      return out;
    },
  };
}

/** Predict from raw (unstandardized) feature rows. */
export interface RowRegressor {
  predict(xsRaw: number[]): number;
}

/** Binary classifier over raw (unstandardized) feature rows. */
export interface RowClassifier {
  /** Hard class prediction (0/1). */
  predict(xsRaw: number[]): 0 | 1;
  /** Probability of class 1 when the model exposes one. */
  predictProba?(xsRaw: number[]): number;
}

/** Normalize non-negative importances to sum 1 (uniform fallback when the
 *  model produced no splits at all). */
export function normalizeImportance(raw: number[]): number[] {
  const total = raw.reduce((a, b) => a + Math.max(b, 0), 0);
  if (total <= 1e-12) return raw.map(() => 1 / Math.max(raw.length, 1));
  return raw.map((v) => Math.max(v, 0) / total);
}

/* ------------------------------------------------------------------ */
/* Lasso / elastic net — coordinate descent with soft-thresholding     */
/* ------------------------------------------------------------------ */

/**
 * Elastic-net linear regression fit by cyclic coordinate descent on the
 * sklearn-convention objective
 *   (1/2n)·‖y − b − X·w‖² + α·(l1Ratio·‖w‖₁ + ½(1−l1Ratio)·‖w‖²),
 * so `alpha`/`l1Ratio` mean exactly what they mean in sklearn's
 * Lasso/ElasticNet. Features are standardized with train statistics; the
 * intercept is never penalized. l1Ratio = 1 reduces to lasso.
 */
export function fitElasticNet(
  X: number[][],
  y: number[],
  opts: { alpha?: number; l1Ratio?: number; sweeps?: number } = {}
): RowRegressor & { std: Standardizer; w: number[]; b: number } {
  const alpha = opts.alpha ?? 1;
  const l1Ratio = opts.l1Ratio ?? 0.5;
  const sweeps = opts.sweeps ?? 600;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));

  const yMean = mean(y);
  const w = new Array<number>(k).fill(0);
  // Column means of Z (≈ 0, kept for an exact intercept refresh).
  const zMeans = new Array<number>(k).fill(0);
  const zSq = new Array<number>(k).fill(0); // (1/n)·Σ z_j²
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const v = Z[i]![j] ?? 0;
      zMeans[j] = (zMeans[j] ?? 0) + v / n;
      zSq[j] = (zSq[j] ?? 0) + (v * v) / n;
    }
  }

  // Residual r = y − b − Z·w, maintained incrementally.
  let b = yMean;
  const r = y.map((v) => v - b);
  const l1Pen = alpha * l1Ratio;
  const l2Pen = alpha * (1 - l1Ratio);

  for (let s = 0; s < sweeps; s++) {
    for (let j = 0; j < k; j++) {
      const zj = zSq[j] ?? 0;
      const denom = zj + l2Pen;
      if (denom <= 1e-12) continue;
      // rho = (1/n)·Σ z_ij·(r_i + z_ij·w_j) — partial residual correlation.
      let rho = 0;
      for (let i = 0; i < n; i++) {
        rho += (Z[i]![j] ?? 0) * ((r[i] ?? 0) + (Z[i]![j] ?? 0) * w[j]!);
      }
      rho /= n;
      const soft = Math.sign(rho) * Math.max(Math.abs(rho) - l1Pen, 0);
      const wNew = soft / denom;
      if (wNew !== w[j]) {
        const delta = w[j]! - wNew;
        for (let i = 0; i < n; i++) {
          r[i] = (r[i] ?? 0) + (Z[i]![j] ?? 0) * delta;
        }
        w[j] = wNew;
      }
    }
    // Intercept refresh: b = mean(y) − Σ_j w_j·mean(z_j).
    let shift = 0;
    for (let j = 0; j < k; j++) shift += w[j]! * (zMeans[j] ?? 0);
    const bNew = yMean - shift;
    if (bNew !== b) {
      const db = bNew - b;
      for (let i = 0; i < n; i++) r[i] = (r[i] ?? 0) - db;
      b = bNew;
    }
  }

  return {
    std,
    w,
    b,
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      let acc = b;
      for (let j = 0; j < k; j++) acc += w[j]! * (z[j] ?? 0);
      return acc;
    },
  };
}

/* ------------------------------------------------------------------ */
/* k-nearest neighbours                                                */
/* ------------------------------------------------------------------ */

/** Euclidean distance in standardized space to every stored train row. */
function knnNeighbours(Z: number[][], z: number[]): number[] {
  const dists: number[] = [];
  for (let i = 0; i < Z.length; i++) {
    let d = 0;
    const row = Z[i]!;
    for (let j = 0; j < z.length; j++) {
      const diff = (row[j] ?? 0) - (z[j] ?? 0);
      d += diff * diff;
    }
    dists.push(d);
  }
  return dists;
}

/** Indices of the k smallest distances (stable on ties → train order). */
function kSmallest(dists: number[], k: number): number[] {
  const order = dists.map((d, i) => ({ d, i }));
  order.sort((a, b) => (a.d === b.d ? a.i - b.i : a.d - b.d));
  return order.slice(0, Math.min(k, order.length)).map((o) => o.i);
}

/** kNN regression: mean label of the k nearest train points. */
export function fitKnnRegressor(
  X: number[][],
  y: number[],
  opts: { k?: number } = {}
): RowRegressor {
  const k = opts.k ?? 5;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  return {
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      const idx = kSmallest(knnNeighbours(Z, z), k);
      return mean(idx.map((i) => y[i] ?? 0));
    },
  };
}

/* ------------------------------------------------------------------ */
/* CART decision trees (variance + gini)                               */
/* ------------------------------------------------------------------ */

interface CartNode {
  n: number;
  feature?: number;
  threshold?: number;
  left?: CartNode;
  right?: CartNode;
  /** Regression: mean y · Classification: P(class 1). */
  value?: number;
}

/** Sum of squared errors of y over the given row indices. */
function sseOf(y: number[], idx: number[]): { sse: number; mean: number } {
  let s = 0;
  let sq = 0;
  for (const i of idx) {
    const v = y[i] ?? 0;
    s += v;
    sq += v * v;
  }
  const m = s / idx.length;
  return { sse: sq - (s * s) / idx.length, mean: m };
}

/**
 * Best CART split over row indices: scan every feature in column order,
 * every midpoint between consecutive distinct sorted values, and keep the
 * FIRST strict maximum of the variance-reduction gain (sklearn's
 * BestSplitter tie convention). Returns the split plus its SSE reduction,
 * or null when no positive-gain split exists.
 */
function bestVarianceSplit(
  X: number[][],
  y: number[],
  idx: number[]
): { feature: number; threshold: number; gain: number } | null {
  const n = idx.length;
  const k = X[0]?.length ?? 0;
  let sumAll = 0;
  let sqAll = 0;
  for (const i of idx) {
    const v = y[i] ?? 0;
    sumAll += v;
    sqAll += v * v;
  }
  const sseP = sqAll - (sumAll * sumAll) / n;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestGain = 1e-12; // strictly positive improvement required
  for (let j = 0; j < k; j++) {
    const order = [...idx].sort((a, b) => (X[a]![j] ?? 0) - (X[b]![j] ?? 0));
    let sumL = 0;
    let sqL = 0;
    for (let m = 0; m + 1 < n; m++) {
      const v = y[order[m]!] ?? 0;
      sumL += v;
      sqL += v * v;
      const xCur = X[order[m]!]![j] ?? 0;
      const xNext = X[order[m + 1]!]![j] ?? 0;
      if (xCur === xNext) continue; // no midpoint between equal values
      const nL = m + 1;
      const nR = n - nL;
      const sseL = sqL - (sumL * sumL) / nL;
      const sumR = sumAll - sumL;
      const sqR = sqAll - sqL;
      const sseR = sqR - (sumR * sumR) / nR;
      const gain = sseP - sseL - sseR;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = j;
        bestThreshold = (xCur + xNext) / 2;
      }
    }
  }
  return bestFeature >= 0
    ? { feature: bestFeature, threshold: bestThreshold, gain: bestGain }
    : null;
}

/** Build a variance-reduction CART subtree over `idx` (z-space rows).
 *  `gains[j]` accumulates (SSE_p − SSE_l − SSE_r)/N per split, sklearn's
 *  feature-importance convention. */
function buildRegressionNode(
  X: number[][],
  y: number[],
  idx: number[],
  depth: number,
  maxDepth: number,
  gains: number[]
): CartNode {
  const { sse, mean: value } = sseOf(y, idx);
  const node: CartNode = { n: idx.length };
  if (depth >= maxDepth || idx.length < 2 || sse <= 1e-12) {
    node.value = value;
    return node;
  }
  const split = bestVarianceSplit(X, y, idx);
  if (!split) {
    node.value = value;
    return node;
  }
  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const i of idx) {
    if ((X[i]![split.feature] ?? 0) <= split.threshold) leftIdx.push(i);
    else rightIdx.push(i);
  }
  if (leftIdx.length === 0 || rightIdx.length === 0) {
    node.value = value;
    return node;
  }
  gains[split.feature] = (gains[split.feature] ?? 0) + split.gain / y.length;
  node.feature = split.feature;
  node.threshold = split.threshold;
  node.left = buildRegressionNode(X, y, leftIdx, depth + 1, maxDepth, gains);
  node.right = buildRegressionNode(X, y, rightIdx, depth + 1, maxDepth, gains);
  return node;
}

function predictTreeNode(node: CartNode, z: number[]): number {
  let cur = node;
  while (cur.value === undefined) {
    const goLeft = (z[cur.feature ?? 0] ?? 0) <= (cur.threshold ?? 0);
    cur = (goLeft ? cur.left : cur.right) ?? cur;
  }
  return cur.value ?? 0;
}

/** CART regression tree with depth cap; importance = raw variance-reduction
 *  gains (normalize with normalizeImportance for display). */
export function fitRegressionTree(
  X: number[][],
  y: number[],
  opts: { maxDepth?: number } = {}
): RowRegressor & { rawImportance: number[] } {
  const maxDepth = opts.maxDepth ?? 4;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const idx = Array.from({ length: X.length }, (_, i) => i);
  const gains = new Array<number>(Z[0]?.length ?? 0).fill(0);
  const root = buildRegressionNode(Z, y, idx, 0, maxDepth, gains);
  return {
    rawImportance: gains,
    predict(xsRaw: number[]) {
      return predictTreeNode(root, std.transform(xsRaw));
    },
  };
}

/* ------------------------------------------------------------------ */
/* Bootstrap-aggregated forests (regression)                           */
/* ------------------------------------------------------------------ */

/** Random forest regression: seeded bootstrap aggregation of depth-capped
 *  CART trees. Importance = mean of per-tree normalized importances,
 *  renormalized (sklearn's convention). */
export function fitRandomForestRegressor(
  X: number[][],
  y: number[],
  opts: { nEstimators?: number; maxDepth?: number; rand: () => number } = { rand: mulberry32(0) }
): RowRegressor & { importance: number[] } {
  const nEstimators = opts.nEstimators ?? 50;
  const maxDepth = opts.maxDepth ?? 6;
  const rand = opts.rand;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const trees: CartNode[] = [];
  const perTreeImportance: number[][] = [];
  for (let t = 0; t < nEstimators; t++) {
    const boot: number[] = [];
    for (let i = 0; i < n; i++) boot.push(Math.floor(rand() * n));
    const gains = new Array<number>(k).fill(0);
    trees.push(buildRegressionNode(Z, y, boot, 0, maxDepth, gains));
    perTreeImportance.push(normalizeImportance(gains));
  }
  const importance = normalizeImportance(
    perTreeImportance.reduce(
      (acc, imp) => acc.map((v, j) => v + (imp[j] ?? 0)),
      new Array<number>(k).fill(0)
    )
  );
  return {
    importance,
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      return mean(trees.map((root) => predictTreeNode(root, z)));
    },
  };
}

/* ------------------------------------------------------------------ */
/* Gradient boosting (regression)                                      */
/* ------------------------------------------------------------------ */

/** Gradient-boosted regression: depth-3 CART trees fit to the running
 *  residuals, each shrunk by `learningRate`. Importance accumulates raw
 *  variance-reduction gains across all rounds, normalized to Σ = 1. */
export function fitGradientBoostingRegressor(
  X: number[][],
  y: number[],
  opts: { rounds?: number; learningRate?: number; maxDepth?: number } = {}
): RowRegressor & { importance: number[] } {
  const rounds = opts.rounds ?? 50;
  const lr = opts.learningRate ?? 0.1;
  const maxDepth = opts.maxDepth ?? 3;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const f0 = mean(y);
  const F = new Array<number>(n).fill(f0);
  const trees: Array<{ root: CartNode }> = [];
  const gains = new Array<number>(k).fill(0);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let r = 0; r < rounds; r++) {
    const residual = F.map((f, i) => (y[i] ?? 0) - f);
    const root = buildRegressionNode(Z, residual, idx, 0, maxDepth, gains);
    trees.push({ root });
    for (let i = 0; i < n; i++) {
      F[i] = (F[i] ?? 0) + lr * predictTreeNode(root, Z[i]!);
    }
  }
  return {
    importance: normalizeImportance(gains),
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      let acc = f0;
      for (const t of trees) acc += lr * predictTreeNode(t.root, z);
      return acc;
    },
  };
}

/* ------------------------------------------------------------------ */
/* MLP — one tanh hidden layer (regression + classification)           */
/* ------------------------------------------------------------------ */

interface MlpWeights {
  /** [hidden][k+1] — last column is the bias. */
  W1: number[][];
  /** [1][hidden+1] — last column is the bias. */
  W2: number[][];
}

function initMlp(k: number, hidden: number, rand: () => number): MlpWeights {
  const scale1 = Math.sqrt(1 / Math.max(k, 1));
  const scale2 = Math.sqrt(1 / Math.max(hidden, 1));
  const W1 = Array.from({ length: hidden }, () =>
    Array.from({ length: k + 1 }, (_, j) => (j === k ? 0 : gaussian(rand) * scale1))
  );
  const W2 = [Array.from({ length: hidden + 1 }, (_, h) => (h === hidden ? 0 : gaussian(rand) * scale2))];
  return { W1, W2 };
}

/** Regression MLP: tanh hidden layer, linear output, full-batch gradient
 *  descent on MSE. Targets are standardized with train statistics and
 *  mapped back at predict time (pIC50-scale targets would otherwise crawl
 *  — the same bug class the toolkit's neural.py guards against). */
export function fitMlpRegressor(
  X: number[][],
  y: number[],
  opts: { hidden?: number; epochs?: number; lr?: number; rand: () => number } = { rand: mulberry32(0) }
): RowRegressor {
  const hidden = opts.hidden ?? 16;
  const epochs = opts.epochs ?? 200;
  const lr = opts.lr ?? 0.05;
  const rand = opts.rand;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const yStats = stats(y);
  const t = y.map((v) => (v - yStats.mu) / yStats.sigma);

  const { W1, W2 } = initMlp(k, hidden, rand);
  const a1 = new Array<number>(hidden).fill(0);

  for (let e = 0; e < epochs; e++) {
    const gW1 = Array.from({ length: hidden }, () => new Array<number>(k + 1).fill(0));
    const gW2 = [new Array<number>(hidden + 1).fill(0)];
    for (let i = 0; i < n; i++) {
      const zrow = Z[i]!;
      // forward
      for (let h = 0; h < hidden; h++) {
        let acc = W1[h]![k]!;
        for (let j = 0; j < k; j++) acc += W1[h]![j]! * (zrow[j] ?? 0);
        a1[h] = Math.tanh(acc);
      }
      let out = W2[0]![hidden]!;
      for (let h = 0; h < hidden; h++) out += W2[0]![h]! * a1[h]!;
      const err = (out - (t[i] ?? 0)) / n; // d(mean squared error)/d(out)
      // backward
      for (let h = 0; h < hidden; h++) {
        gW2[0]![h]! += err * a1[h]!;
        const dz = (1 - a1[h]! * a1[h]!) * W2[0]![h]! * err;
        for (let j = 0; j < k; j++) gW1[h]![j]! += dz * (zrow[j] ?? 0);
        gW1[h]![k]! += dz;
      }
      gW2[0]![hidden]! += err;
    }
    for (let h = 0; h < hidden; h++) {
      for (let j = 0; j <= k; j++) W1[h]![j]! -= lr * gW1[h]![j]!;
    }
    for (let j = 0; j <= hidden; j++) W2[0]![j]! -= lr * gW2[0]![j]!;
  }

  return {
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      for (let h = 0; h < hidden; h++) {
        let acc = W1[h]![k]!;
        for (let j = 0; j < k; j++) acc += W1[h]![j]! * (z[j] ?? 0);
        a1[h] = Math.tanh(acc);
      }
      let out = W2[0]![hidden]!;
      for (let h = 0; h < hidden; h++) out += W2[0]![h]! * a1[h]!;
      return out * yStats.sigma + yStats.mu;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Classification: kNN / CART gini / forest / NB / SVM / MLP           */
/* ------------------------------------------------------------------ */

/** kNN classifier: probability = fraction of class-1 neighbours among the
 *  k nearest train points (odd k ⇒ no vote ties). */
export function fitKnnClassifier(
  X: number[][],
  labels: number[],
  opts: { k?: number } = {}
): RowClassifier {
  const k = opts.k ?? 5;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  return {
    predictProba(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      const idx = kSmallest(knnNeighbours(Z, z), k);
      return mean(idx.map((i) => labels[i] ?? 0));
    },
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      const idx = kSmallest(knnNeighbours(Z, z), k);
      return mean(idx.map((i) => labels[i] ?? 0)) >= 0.5 ? 1 : 0;
    },
  };
}

/** Gini impurity of a 0/1 label subset. */
function giniOf(labels: number[], idx: number[]): number {
  let ones = 0;
  for (const i of idx) if ((labels[i] ?? 0) === 1) ones++;
  const p = ones / idx.length;
  return 1 - p * p - (1 - p) * (1 - p);
}

/** Best gini split — same scan/tie convention as bestVarianceSplit.
 *  Gain = n·gini_p − n_l·gini_l − n_r·gini_r. */
function bestGiniSplit(
  X: number[][],
  labels: number[],
  idx: number[]
): { feature: number; threshold: number; gain: number } | null {
  const n = idx.length;
  const k = X[0]?.length ?? 0;
  const giniP = giniOf(labels, idx);
  let onesAll = 0;
  for (const i of idx) if ((labels[i] ?? 0) === 1) onesAll++;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestGain = 1e-12;
  for (let j = 0; j < k; j++) {
    const order = [...idx].sort((a, b) => (X[a]![j] ?? 0) - (X[b]![j] ?? 0));
    let onesL = 0;
    for (let m = 0; m + 1 < n; m++) {
      if ((labels[order[m]!] ?? 0) === 1) onesL++;
      const xCur = X[order[m]!]![j] ?? 0;
      const xNext = X[order[m + 1]!]![j] ?? 0;
      if (xCur === xNext) continue;
      const nL = m + 1;
      const nR = n - nL;
      const pL = onesL / nL;
      const giniL = 1 - pL * pL - (1 - pL) * (1 - pL);
      const onesR = onesAll - onesL;
      const pR = onesR / nR;
      const giniR = 1 - pR * pR - (1 - pR) * (1 - pR);
      const gain = n * giniP - nL * giniL - nR * giniR;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = j;
        bestThreshold = (xCur + xNext) / 2;
      }
    }
  }
  return bestFeature >= 0
    ? { feature: bestFeature, threshold: bestThreshold, gain: bestGain }
    : null;
}

function countOnes(labels: number[], idx: number[]): number {
  let ones = 0;
  for (const i of idx) if ((labels[i] ?? 0) === 1) ones++;
  return ones;
}

function buildClassificationNode(
  X: number[][],
  labels: number[],
  idx: number[],
  depth: number,
  maxDepth: number,
  gains: number[]
): CartNode {
  const ones = countOnes(labels, idx);
  const value = ones / idx.length;
  const node: CartNode = { n: idx.length };
  if (depth >= maxDepth || idx.length < 2 || value === 0 || value === 1) {
    node.value = value;
    return node;
  }
  const split = bestGiniSplit(X, labels, idx);
  if (!split) {
    node.value = value;
    return node;
  }
  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const i of idx) {
    if ((X[i]![split.feature] ?? 0) <= split.threshold) leftIdx.push(i);
    else rightIdx.push(i);
  }
  if (leftIdx.length === 0 || rightIdx.length === 0) {
    node.value = value;
    return node;
  }
  gains[split.feature] = (gains[split.feature] ?? 0) + split.gain / labels.length;
  node.feature = split.feature;
  node.threshold = split.threshold;
  node.left = buildClassificationNode(X, labels, leftIdx, depth + 1, maxDepth, gains);
  node.right = buildClassificationNode(X, labels, rightIdx, depth + 1, maxDepth, gains);
  return node;
}

/** CART classification tree (gini). Leaves carry P(class 1); hard
 *  prediction thresholds that probability at 0.5. */
export function fitClassificationTree(
  X: number[][],
  labels: number[],
  opts: { maxDepth?: number } = {}
): RowClassifier & { rawImportance: number[] } {
  const maxDepth = opts.maxDepth ?? 4;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const idx = Array.from({ length: X.length }, (_, i) => i);
  const gains = new Array<number>(Z[0]?.length ?? 0).fill(0);
  const root = buildClassificationNode(Z, labels, idx, 0, maxDepth, gains);
  const proba = (xsRaw: number[]): number => predictTreeNode(root, std.transform(xsRaw));
  return {
    rawImportance: gains,
    predictProba: proba,
    predict(xsRaw: number[]) {
      return proba(xsRaw) >= 0.5 ? 1 : 0;
    },
  };
}

/** Random forest classifier: seeded bagging of gini CART trees;
 *  probability = mean of per-tree leaf class fractions. */
export function fitRandomForestClassifier(
  X: number[][],
  labels: number[],
  opts: { nEstimators?: number; maxDepth?: number; rand: () => number } = { rand: mulberry32(0) }
): RowClassifier & { importance: number[] } {
  const nEstimators = opts.nEstimators ?? 50;
  const maxDepth = opts.maxDepth ?? 6;
  const rand = opts.rand;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const trees: CartNode[] = [];
  const perTreeImportance: number[][] = [];
  for (let t = 0; t < nEstimators; t++) {
    const boot: number[] = [];
    for (let i = 0; i < n; i++) boot.push(Math.floor(rand() * n));
    const gains = new Array<number>(k).fill(0);
    trees.push(buildClassificationNode(Z, labels, boot, 0, maxDepth, gains));
    perTreeImportance.push(normalizeImportance(gains));
  }
  const importance = normalizeImportance(
    perTreeImportance.reduce(
      (acc, imp) => acc.map((v, j) => v + (imp[j] ?? 0)),
      new Array<number>(k).fill(0)
    )
  );
  const proba = (xsRaw: number[]): number => {
    const z = std.transform(xsRaw);
    return mean(trees.map((root) => predictTreeNode(root, z)));
  };
  return {
    importance,
    predictProba: proba,
    predict(xsRaw: number[]) {
      return proba(xsRaw) >= 0.5 ? 1 : 0;
    },
  };
}

/** Gaussian Naive Bayes: per-class Gaussian log-posterior with sklearn's
 *  var_smoothing epsilon (1e-9 × the largest global feature variance)
 *  added to every class-conditional variance. */
export function fitGaussianNB(
  X: number[][],
  labels: number[],
  opts: { varSmoothing?: number } = {}
): RowClassifier & { priors: [number, number]; logPosterior(xsRaw: number[]): [number, number] } {
  const varSmoothing = opts.varSmoothing ?? 1e-9;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));

  // sklearn's epsilon: var_smoothing × max over features of the global
  // (whole-train-set) population variance.
  let maxVar = 0;
  for (let j = 0; j < k; j++) {
    const s = stats(Z.map((row) => row[j] ?? 0));
    if (s.sigma * s.sigma > maxVar) maxVar = s.sigma * s.sigma;
  }
  const eps = varSmoothing * maxVar;

  const fit: Array<{ prior: number; mus: number[]; vars: number[] }> = [];
  for (const c of [0, 1] as const) {
    const rows = Z.filter((_, i) => (labels[i] ?? 0) === c);
    const prior = rows.length / n;
    const mus: number[] = [];
    const vars: number[] = [];
    for (let j = 0; j < k; j++) {
      const s = stats(rows.map((row) => row[j] ?? 0));
      mus.push(s.mu);
      vars.push(s.sigma * s.sigma + eps);
    }
    fit.push({ prior, mus, vars });
  }

  function logPosterior(xsRaw: number[]): [number, number] {
    const z = std.transform(xsRaw);
    const out: [number, number] = [0, 0];
    for (let c = 0 as 0 | 1; c <= 1; c++) {
      const f = fit[c]!;
      let lp = Math.log(Math.max(f.prior, 1e-300));
      for (let j = 0; j < k; j++) {
        const v = f.vars[j]!;
        const diff = (z[j] ?? 0) - f.mus[j]!;
        lp += -0.5 * Math.log(2 * Math.PI * v) - (diff * diff) / (2 * v);
      }
      out[c] = lp;
    }
    return out;
  }

  return {
    priors: [fit[0]!.prior, fit[1]!.prior],
    logPosterior,
    predictProba(xsRaw: number[]) {
      const [lp0, lp1] = logPosterior(xsRaw);
      return 1 / (1 + Math.exp(lp0 - lp1));
    },
    predict(xsRaw: number[]) {
      const [lp0, lp1] = logPosterior(xsRaw);
      return lp1 > lp0 ? 1 : 0; // argmax with ties → class 0 (np.argmax)
    },
  };
}

/** Linear SVM: L2-regularized squared-hinge objective solved by batch
 *  gradient descent,
 *    J(θ) = ½‖θ‖² + C·Σ max(0, 1 − y·θ·u)ᵀ²,  u = [z, 1],
 *  mirroring liblinear/LinearSVC — including its quirk of regularizing the
 *  bias (a constant-1 feature). The step size is 1/L with L an explicit
 *  Lipschitz upper bound (1 + 2C·Σ‖u‖²), so GD never diverges. */
export function fitLinearSvm(
  X: number[][],
  labels: number[],
  opts: { C?: number; epochs?: number } = {}
): RowClassifier & { w: number[]; bias: number } {
  const C = opts.C ?? 1;
  const epochs = opts.epochs ?? 3000;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));

  const theta = new Array<number>(k + 1).fill(0); // [w…, b]
  let lipSum = 0;
  for (let i = 0; i < n; i++) {
    let s = 1; // bias feature contributes 1
    for (let j = 0; j < k; j++) s += (Z[i]![j] ?? 0) ** 2;
    lipSum += s;
  }
  const step = 1 / (1 + 2 * C * lipSum);

  for (let e = 0; e < epochs; e++) {
    const grad = new Array<number>(k + 1).fill(0);
    for (let i = 0; i < n; i++) {
      let margin = theta[k]!;
      for (let j = 0; j < k; j++) margin += theta[j]! * (Z[i]![j] ?? 0);
      const yi = (labels[i] ?? 0) === 1 ? 1 : -1;
      const viol = 1 - yi * margin;
      if (viol > 0) {
        const coef = 2 * C * viol * yi;
        for (let j = 0; j < k; j++) grad[j]! -= coef * (Z[i]![j] ?? 0);
        grad[k]! -= coef;
      }
    }
    for (let j = 0; j <= k; j++) {
      // ∇J = θ + (−2C·Σ viol·y·u); the loop above accumulated the hinge part.
      theta[j]! -= step * (theta[j]! + grad[j]!);
    }
  }

  const w = theta.slice(0, k);
  const bias = theta[k] ?? 0;
  return {
    w,
    bias,
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      let margin = bias;
      for (let j = 0; j < k; j++) margin += w[j]! * (z[j] ?? 0);
      return margin >= 0 ? 1 : 0;
    },
  };
}

/** Classification MLP: tanh hidden layer, sigmoid output, full-batch
 *  gradient descent on cross-entropy. Seeded initialization. */
export function fitMlpClassifier(
  X: number[][],
  labels: number[],
  opts: { hidden?: number; epochs?: number; lr?: number; rand: () => number } = { rand: mulberry32(0) }
): RowClassifier {
  const hidden = opts.hidden ?? 16;
  const epochs = opts.epochs ?? 200;
  const lr = opts.lr ?? 0.05;
  const rand = opts.rand;
  const n = X.length;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));

  const { W1, W2 } = initMlp(k, hidden, rand);
  const a1 = new Array<number>(hidden).fill(0);

  for (let e = 0; e < epochs; e++) {
    const gW1 = Array.from({ length: hidden }, () => new Array<number>(k + 1).fill(0));
    const gW2 = [new Array<number>(hidden + 1).fill(0)];
    for (let i = 0; i < n; i++) {
      const zrow = Z[i]!;
      for (let h = 0; h < hidden; h++) {
        let acc = W1[h]![k]!;
        for (let j = 0; j < k; j++) acc += W1[h]![j]! * (zrow[j] ?? 0);
        a1[h] = Math.tanh(acc);
      }
      let out = W2[0]![hidden]!;
      for (let h = 0; h < hidden; h++) out += W2[0]![h]! * a1[h]!;
      const p = 1 / (1 + Math.exp(-out));
      const err = (p - (labels[i] ?? 0)) / n; // d(mean CE)/d(out)
      for (let h = 0; h < hidden; h++) {
        gW2[0]![h]! += err * a1[h]!;
        const dz = (1 - a1[h]! * a1[h]!) * W2[0]![h]! * err;
        for (let j = 0; j < k; j++) gW1[h]![j]! += dz * (zrow[j] ?? 0);
        gW1[h]![k]! += dz;
      }
      gW2[0]![hidden]! += err;
    }
    for (let h = 0; h < hidden; h++) {
      for (let j = 0; j <= k; j++) W1[h]![j]! -= lr * gW1[h]![j]!;
    }
    for (let j = 0; j <= hidden; j++) W2[0]![j]! -= lr * gW2[0]![j]!;
  }

  const proba = (xsRaw: number[]): number => {
    const z = std.transform(xsRaw);
    for (let h = 0; h < hidden; h++) {
      let acc = W1[h]![k]!;
      for (let j = 0; j < k; j++) acc += W1[h]![j]! * (z[j] ?? 0);
      a1[h] = Math.tanh(acc);
    }
    let out = W2[0]![hidden]!;
    for (let h = 0; h < hidden; h++) out += W2[0]![h]! * a1[h]!;
    return 1 / (1 + Math.exp(-out));
  };
  return {
    predictProba: proba,
    predict(xsRaw: number[]) {
      return proba(xsRaw) >= 0.5 ? 1 : 0;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Hierarchical SVR (HSVR) — seeded k-means + local ridge blend        */
/* ------------------------------------------------------------------ */

/** Result of a seeded k-means run (Lloyd iterations in the clustered space). */
export interface KMeansFit {
  /** Cluster index per row (0..k−1; ties break to the lowest index). */
  assignments: number[];
  /** Converged centroids. */
  centroids: number[][];
  /** Member row indices per cluster. */
  members: number[][];
  /** Lloyd iterations until assignments stopped changing. */
  iterations: number;
}

/**
 * Deterministic seeded k-means: k-means++ seeding driven by `rand`, then
 * Lloyd iterations until the assignment vector stops changing (capped at
 * 100). Distance ties break to the lowest cluster index; an empty cluster
 * re-seeds at the row currently farthest from its assigned centroid.
 */
export function kMeansSeeded(Z: number[][], k: number, rand: () => number): KMeansFit {
  const n = Z.length;
  const kk = Math.max(1, Math.min(k, n));
  const dims = Z[0]?.length ?? 0;

  const dist2 = (a: number[], b: number[]): number => {
    let d = 0;
    for (let j = 0; j < dims; j++) d += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2;
    return d;
  };

  /* k-means++ seeding. */
  const centroids: number[][] = [(Z[Math.floor(rand() * n)] ?? Z[0]!).slice()];
  while (centroids.length < kk) {
    const d2 = Z.map((row) => Math.min(...centroids.map((c) => dist2(row, c))));
    const total = d2.reduce((a, b) => a + b, 0);
    if (total <= 1e-12) {
      centroids.push((Z[0] ?? []).slice()); // degenerate: duplicate points
      continue;
    }
    let r = rand() * total;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      r -= d2[i] ?? 0;
      if (r <= 0) {
        pick = i;
        break;
      }
    }
    centroids.push((Z[pick] ?? Z[0]!).slice());
  }

  /* Lloyd iterations. */
  const assignments = new Array<number>(n).fill(0);
  let iterations = 0;
  for (;;) {
    iterations++;
    let changed = false;
    for (let i = 0; i < n; i++) {
      const row = Z[i]!;
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < kk; c++) {
        const d = dist2(row, centroids[c]!);
        if (d < bestD - 1e-15) {
          bestD = d;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }

    const sums = Array.from({ length: kk }, () => new Array<number>(dims).fill(0));
    const counts = new Array<number>(kk).fill(0);
    for (let i = 0; i < n; i++) {
      const c = assignments[i]!;
      counts[c] = (counts[c] ?? 0) + 1;
      for (let j = 0; j < dims; j++) sums[c]![j] = (sums[c]![j] ?? 0) + (Z[i]![j] ?? 0);
    }
    for (let c = 0; c < kk; c++) {
      if ((counts[c] ?? 0) > 0) {
        centroids[c] = (sums[c] ?? []).map((v) => v / (counts[c] ?? 1));
      } else {
        /* Empty cluster: re-seed at the row farthest from its centroid. */
        let far = 0;
        let farD = -1;
        for (let i = 0; i < n; i++) {
          const d = dist2(Z[i]!, centroids[assignments[i] ?? 0]!);
          if (d > farD) {
            farD = d;
            far = i;
          }
        }
        centroids[c] = (Z[far] ?? Z[0]!).slice();
      }
    }
    if (!changed || iterations >= 100) break;
  }

  const members: number[][] = Array.from({ length: kk }, () => []);
  for (let i = 0; i < n; i++) members[assignments[i] ?? 0]!.push(i);
  return { assignments, centroids, members, iterations };
}

/** Per-cluster diagnostics surfaced by the HSVR fit. */
export interface HsvrClusterInfo {
  /** Train rows assigned to the cluster. */
  size: number;
  /** Trust radius R = radiusFactor × the max member distance to the centroid. */
  radius: number;
  /** In-sample R² of the cluster-local model (constant clusters → 1). */
  localR2: number;
  /** True when the cluster kept too few rows for a ridge fit and fell back
   *  to predicting the cluster mean (the constant stand-in). */
  constant: boolean;
}

/** One train row in the trust-radius scatter's plot coordinates. */
export interface HsvrScatterPoint {
  /** First standardized feature dim. */
  a: number;
  /** Second dim: the raw target (single-feature models) or the second
   *  standardized feature. */
  b: number;
  /** 0-based cluster assignment (the k-means owner). */
  cluster: number;
}

/** The trust-radius scatter geometry: train rows + centroids, projected
 *  onto the scatter's two dims so the UI never recomputes k-means. */
export interface HsvrScatter {
  points: HsvrScatterPoint[];
  /** Centroids in plot coords (b = the cluster's mean target for
   *  single-feature models) with their trust radii. */
  centroids: Array<{ a: number; b: number; radius: number }>;
}

/** The full HSVR model — global ridge + per-cluster local ridges blended
 *  by a linear trust-radius fade. */
export interface HsvrModel extends RowRegressor {
  std: Standardizer;
  /** Per-cluster geometry in standardized space. */
  clusters: Array<{ centroid: number[]; radius: number }>;
  /** Honest diagnostics for the learned-parameters card. */
  info: HsvrClusterInfo[];
  /** The trust-radius scatter geometry (train rows + centroids). */
  scatter: HsvrScatter;
}

/**
 * Hierarchical SVR (the web port of pharmaml.ml.hierarchical.HierarchicalSVR
 * with ridge as the local/global learner): seeded k-means in standardized
 * feature space, one ridge model per cluster plus a global ridge over all
 * rows. Predictions blend the nearest cluster's local model with the global
 * model through a linear trust radius — full local weight at the centroid,
 * zero at R = radiusFactor × the cluster's max member distance; queries
 * beyond R fall back to the pure global prediction. Clusters too small to
 * fit ridge (n < k + 2) keep a constant stand-in at the cluster mean.
 */
export function fitHsvr(
  X: number[][],
  y: number[],
  opts: {
    nClusters?: number;
    radiusFactor?: number;
    alpha?: number;
    rand: () => number;
  } = { rand: mulberry32(0) }
): HsvrModel {
  const nClusters = opts.nClusters ?? 3;
  const radiusFactor = opts.radiusFactor ?? 1.5;
  const alpha = opts.alpha ?? 1;
  const rand = opts.rand;
  const k = X[0]?.length ?? 0;
  const std = fitStandardizer(X);
  const Z = X.map((row) => std.transform(row));
  const km = kMeansSeeded(Z, nClusters, rand);

  /* Global ridge over every row — also the far-query fallback. */
  const global = fitMultivariateRegression(X, y, { alpha });

  const clusters: Array<{ centroid: number[]; radius: number }> = [];
  const locals: Array<RowRegressor> = [];
  const info: HsvrClusterInfo[] = [];
  /* Centroids in plot coords for the trust-radius scatter (b = the
   * cluster's mean target — the local model's constant stand-in). */
  const scatterCentroids: Array<{ a: number; b: number; radius: number }> = [];
  const singleFeature = k <= 1;

  km.members.forEach((members, c) => {
    const centroid = km.centroids[c]!;
    let maxDist = 0;
    for (const i of members) {
      const d = Math.sqrt(
        Z[i]!.reduce((acc, v, j) => acc + (v - (centroid[j] ?? 0)) ** 2, 0)
      );
      if (d > maxDist) maxDist = d;
    }
    /* Strictly positive so weight at the centroid is exactly 1. */
    const radius = maxDist * radiusFactor + 1e-9;
    clusters.push({ centroid: [...centroid], radius });

    const localX = members.map((i) => X[i]!);
    const localY = members.map((i) => y[i]!);
    scatterCentroids.push({
      a: centroid[0] ?? 0,
      b: singleFeature ? mean(localY) : (centroid[1] ?? 0),
      radius,
    });
    if (localX.length >= k + 2 && localX.length > 0) {
      const m = fitMultivariateRegression(localX, localY, { alpha });
      locals.push({ predict: (xs) => m.predict(xs) });
      const pred = localX.map((xs) => m.predict(xs));
      info.push({
        size: members.length,
        radius,
        localR2: regressionMetrics(localY, pred).r2,
        constant: false,
      });
    } else {
      /* Constant stand-in (the _ConstantSVR equivalent): cluster mean. */
      const yMean = mean(localY);
      locals.push({ predict: () => yMean });
      const sst = localY.reduce((acc, v) => acc + (v - yMean) ** 2, 0);
      info.push({
        size: members.length,
        radius,
        localR2: sst <= 1e-12 ? 1 : 0,
        constant: true,
      });
    }
  });

  /* The trust-radius scatter: train rows in (z1, target) or (z1, z2)
   *  coords with their k-means assignments — the geometry the UI plots
   *  without ever re-running the clustering. */
  const scatter: HsvrScatter = {
    points: Z.map((zRow, i) => ({
      a: zRow[0] ?? 0,
      b: singleFeature ? (y[i] ?? 0) : (zRow[1] ?? 0),
      cluster: km.assignments[i] ?? 0,
    })),
    centroids: scatterCentroids,
  };

  return {
    std,
    clusters,
    info,
    scatter,
    predict(xsRaw: number[]) {
      const z = std.transform(xsRaw);
      /* Nearest centroid (ties → lowest index, matching k-means). */
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < clusters.length; c++) {
        let d = 0;
        const cen = clusters[c]!.centroid;
        for (let j = 0; j < z.length; j++) d += ((z[j] ?? 0) - (cen[j] ?? 0)) ** 2;
        if (d < bestD - 1e-15) {
          bestD = d;
          best = c;
        }
      }
      const d = Math.sqrt(bestD);
      const R = clusters[best]!.radius;
      const w = Math.max(0, 1 - d / R);
      const local = locals[best]!.predict(xsRaw);
      const glob = global.predict(xsRaw);
      return w * local + (1 - w) * glob;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Aggregation helpers                                                 */
/* ------------------------------------------------------------------ */

export function mean(values: number[]): number {
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

/** Population standard deviation. */
export function populationStd(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length);
}

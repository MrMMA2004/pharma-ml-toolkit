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

/** y = 0.8x + 0.5 + noise·N(0,1), x ~ U(-3, 3). */
export function makeRegressionData(
  n: number,
  noise: number,
  rand: () => number
): { xs: number[]; ys: number[] } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = rand() * 6 - 3;
    const y = 0.8 * x + 0.5 + noise * gaussian(rand);
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

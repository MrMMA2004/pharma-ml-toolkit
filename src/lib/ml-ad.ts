/**
 * Applicability-domain analysis — the browser twin of
 * `pharmaml.ml.applicability` (the 2026-10 user upload).
 *
 * The same two questions the Python module asks, in pure TS so the web
 * playground answers them without a subprocess:
 *
 *  1. **Distance domain** — is the query row inside the training
 *     descriptor space? Mean k-nearest-neighbour distance in the
 *     standardised space, thresholded at a percentile of the training
 *     rows' own (self-excluded) mean distances.
 *  2. **Bootstrap uncertainty** — how much does the fitted model's
 *     answer wobble across bootstrap resamples of its training set?
 *     Per-row standard deviation across the ensemble, thresholded at
 *     the same percentile of the training-side spread.
 *
 * Both fold into `reliable = inside & low uncertainty` — the upload's
 * final verdict per compound. Regression models carry the full
 * bootstrap half (numeric predictions); classification models keep the
 * distance verdict only (hard labels have no honest per-row std — the
 * same contract as the Python side).
 */

/* ------------------------------------------------------------------ */
/* Standardisation                                                     */
/* ------------------------------------------------------------------ */

export interface Standardizer {
  mu: number[];
  sigma: number[];
  /** Apply the fitted (mu, sigma) to a raw row. */
  z(row: number[]): number[];
}

/** Population (ddof = 0) mean/std per column — the sklearn convention. */
export function fitStandardizer(rows: number[][]): Standardizer {
  const n = rows.length;
  const width = rows[0]?.length ?? 0;
  const mu = new Array<number>(width).fill(0);
  for (const row of rows) {
    for (let j = 0; j < width; j += 1) mu[j] += row[j] ?? 0;
  }
  for (let j = 0; j < width; j += 1) mu[j] /= n || 1;
  const sigma = new Array<number>(width).fill(1);
  for (let j = 0; j < width; j += 1) {
    let acc = 0;
    for (const row of rows) {
      const d = (row[j] ?? 0) - mu[j];
      acc += d * d;
    }
    const variance = acc / (n || 1);
    sigma[j] = Math.sqrt(variance) || 1; /* zero-variance column -> unit scale */
  }
  return {
    mu,
    sigma,
    z: (row: number[]) => row.map((v, j) => (v - mu[j]) / sigma[j]),
  };
}

/* ------------------------------------------------------------------ */
/* kNN distance domain                                                 */
/* ------------------------------------------------------------------ */

export interface KnnDomain {
  /** Per training row: mean distance to its k nearest (self excluded). */
  trainMean: number[];
  /** Per query row: mean distance to the k nearest training rows. */
  testMean: number[];
  /** Percentile cut of `trainMean` (standardised units). */
  threshold: number;
  /** Query rows at or under the threshold. */
  inside: boolean[];
}

function percentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0];
  /* Linear interpolation on the empirical CDF (numpy's default). */
  const rank = (p / 100) * (sortedValues.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (rank - lo) * (sortedValues[hi] - sortedValues[lo]);
}

function euclidean(a: number[], b: number[]): number {
  let acc = 0;
  for (let j = 0; j < a.length; j += 1) {
    const d = (a[j] ?? 0) - (b[j] ?? 0);
    acc += d * d;
  }
  return Math.sqrt(acc);
}

/**
 * The distance half of the upload's workflow.
 *
 * Training distances EXCLUDE each row's own zero self-match (the k+1
 * nearest are queried and the first dropped) — the deliberate
 * behavioural upgrade documented on the Python side: the upload's
 * thresholds were systematically slightly too tight.
 */
export function knnDomain(
  trainZ: number[][],
  testZ: number[][],
  k: number,
  p: number
): KnnDomain {
  const kTrain = Math.max(1, Math.min(k, trainZ.length - 1));
  const trainMean = trainZ.map((row, i) => {
    /* k+1 nearest neighbours of the row itself, first (self, distance 0)
     * dropped; with duplicates the zero simply ties at the front. */
    const wanted = Math.min(kTrain + 1, trainZ.length);
    const dists = trainZ
      .map((other, j) => (j === i ? 0 : euclidean(row, other)))
      .sort((a, b) => a - b)
      .slice(0, wanted);
    const used = dists.length > 1 ? dists.slice(1) : dists;
    return used.reduce((a, b) => a + b, 0) / (used.length || 1);
  });
  const sortedTrain = [...trainMean].sort((a, b) => a - b);
  const threshold = percentile(sortedTrain, p);
  const testMean = testZ.map((row) => {
    const dists = trainZ
      .map((other) => euclidean(row, other))
      .sort((a, b) => a - b)
      .slice(0, kTrain);
    return dists.reduce((a, b) => a + b, 0) / (kTrain || 1);
  });
  const inside = testMean.map((d) => d <= threshold);
  return { trainMean, testMean, threshold, inside };
}

/* ------------------------------------------------------------------ */
/* Bootstrap resampling                                                */
/* ------------------------------------------------------------------ */

/** Row indices drawn with replacement (sklearn `resample` semantics). */
export function bootstrapIndices(n: number, rand: () => number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    const r = rand();
    const idx = Math.min(n - 1, Math.floor(r * n));
    out[i] = Math.max(0, idx);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* The composed result                                                 */
/* ------------------------------------------------------------------ */

export interface AdRow {
  /** Row label (SMILES / compound id / `held-out #i`). */
  name: string;
  predicted: number | null;
  predictionSd: number | null;
  knnDistance: number;
  insideAd: boolean;
  lowUncertainty: boolean | null;
  reliable: boolean | null;
}

export interface AdSummary {
  k: number;
  percentile: number;
  distanceThreshold: number;
  /** Null when the bootstrap half was skipped (classification / nModels 0). */
  uncertaintyThreshold: number | null;
  nModels: number | null;
  insideCount: number;
  reliableCount: number | null;
  nQuery: number;
  fitSeconds: number;
}

export interface AdOutcome {
  summary: AdSummary;
  rows: AdRow[];
  /** Training-side mean distances (the panel (a) histogram). */
  trainMean: number[];
}

export interface AdOptions {
  XTrain: number[][];
  yTrain: number[];
  XTest: number[][];
  names: string[];
  k: number;
  percentile: number;
  /** 0 = kNN-only quick pass; classification models always take it. */
  nModels: number;
  /** Deterministic member RNG (seeded per member on the caller side). */
  memberRand: (memberIndex: number) => () => number;
  /** Refit the SAME model family on a resampled training set. */
  refit: (XBoot: number[][], yBoot: number[]) => { predict(row: number[]): number };
}

/** The upload's workflow, one call — the browser edition. */
export function applicabilityDomain(opts: AdOptions): AdOutcome {
  const { XTrain, yTrain, XTest, names, k, percentile: pctValue } = opts;
  const scaler = fitStandardizer(XTrain);
  const trainZ = XTrain.map((r) => scaler.z(r));
  const testZ = XTest.map((r) => scaler.z(r));
  const domain = knnDomain(trainZ, testZ, k, pctValue);

  const nModels = Math.max(0, opts.nModels);
  const nQuery = XTest.length;
  const rows: AdRow[] = XTest.map((row, i) => ({
    name: names[i] ?? `compound_${i}`,
    predicted: null,
    predictionSd: null,
    knnDistance: domain.testMean[i],
    insideAd: domain.inside[i],
    lowUncertainty: null,
    reliable: null,
  }));

  if (nModels === 0) {
    return {
      summary: {
        k,
        percentile: pctValue,
        distanceThreshold: domain.threshold,
        uncertaintyThreshold: null,
        nModels: null,
        insideCount: domain.inside.filter(Boolean).length,
        reliableCount: null,
        nQuery,
        fitSeconds: 0,
      },
      rows,
      trainMean: domain.trainMean,
    };
  }

  const started = Date.now();
  const testMatrix: number[][] = [];
  const trainMatrix: number[][] = [];
  for (let member = 0; member < nModels; member += 1) {
    const idx = bootstrapIndices(XTrain.length, opts.memberRand(member));
    const XBoot = idx.map((i) => XTrain[i]);
    const yBoot = idx.map((i) => yTrain[i]);
    const fit = opts.refit(XBoot, yBoot);
    testMatrix.push(XTest.map((r) => fit.predict(r)));
    trainMatrix.push(XTrain.map((r) => fit.predict(r)));
  }
  const fitSeconds = (Date.now() - started) / 1000;

  const columnMean = (m: number[][], j: number) =>
    m.reduce((a, row) => a + (row[j] ?? 0), 0) / (m.length || 1);
  const columnStd = (m: number[][], j: number) => {
    const mu = columnMean(m, j);
    const acc = m.reduce((a, row) => a + ((row[j] ?? 0) - mu) ** 2, 0);
    return Math.sqrt(acc / (m.length || 1));
  };
  const trainStd = XTrain.map((_, j) => columnStd(trainMatrix, j));
  const sortedTrainStd = [...trainStd].sort((a, b) => a - b);
  const uncertaintyThreshold = percentile(sortedTrainStd, pctValue);

  const predMean = XTest.map((_, j) => columnMean(testMatrix, j));
  const predStd = XTest.map((_, j) => columnStd(testMatrix, j));
  let reliableCount = 0;
  for (let j = 0; j < nQuery; j += 1) {
    const low = predStd[j] <= uncertaintyThreshold;
    const reliable = domain.inside[j] && low;
    if (reliable) reliableCount += 1;
    rows[j] = {
      ...rows[j],
      predicted: predMean[j],
      predictionSd: predStd[j],
      lowUncertainty: low,
      reliable,
    };
  }

  return {
    summary: {
      k,
      percentile: pctValue,
      distanceThreshold: domain.threshold,
      uncertaintyThreshold,
      nModels,
      insideCount: domain.inside.filter(Boolean).length,
      reliableCount,
      nQuery,
      fitSeconds,
    },
    rows,
    trainMean: domain.trainMean,
  };
}

/**
 * Shared ML model registry — the single source of truth for the
 * `/api/ml/train` and `/api/ml/tune` routes (and the UI's mirrored model
 * list). Fitting logic, hyperparameter resolution, and the RandomSearch
 * tuning grids all live here so the two routes can never drift apart.
 */

import {
  fitClassificationTree,
  fitElasticNet,
  fitGaussianNB,
  fitGradientBoostingRegressor,
  fitHsvr,
  fitKnnClassifier,
  fitKnnRegressor,
  fitLinearSvm,
  fitLogisticRegression,
  fitMlpClassifier,
  fitMlpRegressor,
  fitMultivariateRegression,
  fitRandomForestClassifier,
  fitRandomForestRegressor,
  fitRegressionTree,
  mulberry32,
  normalizeImportance,
  type HsvrClusterInfo,
} from "@/lib/ml-math";

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

/** Regression models: 3 legacy (byte-compatible) + 8 (incl. hsvr). */
export const REGRESSION_MODELS = [
  "linear",
  "polynomial",
  "ridge",
  "lasso",
  "elastic_net",
  "knn",
  "decision_tree",
  "random_forest",
  "gradient_boosting",
  "mlp",
  "hsvr",
] as const;

/** Classification models: legacy logistic + 6 rebuilt. */
export const CLASSIFICATION_MODELS = [
  "logistic",
  "knn",
  "decision_tree",
  "random_forest",
  "gaussian_nb",
  "svm_linear",
  "mlp",
] as const;

/* ------------------------------------------------------------------ */
/* Hyperparameters                                                     */
/* ------------------------------------------------------------------ */

/** Hyperparameter source (a parsed request body — all fields optional). */
export interface HyperSource {
  alpha?: number;
  l1Ratio?: number;
  l2?: number;
  k?: number;
  maxDepth?: number;
  nEstimators?: number;
  rounds?: number;
  learningRate?: number;
  hidden?: number;
  epochs?: number;
  lr?: number;
  C?: number;
  /** HSVR: k-means cluster count. */
  nClusters?: number;
  /** HSVR: trust radius × the cluster's max member distance. */
  radiusFactor?: number;
}

/** Hyperparameters with per-model defaults applied. */
export interface Hyper {
  alpha: number;
  l1Ratio: number;
  k: number;
  maxDepth: number;
  nEstimators: number;
  rounds: number;
  learningRate: number;
  hidden: number;
  epochs: number;
  lr: number;
  C: number;
  l2: number;
  nClusters: number;
  radiusFactor: number;
}

function num(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function resolveHyper(req: HyperSource): Hyper {
  return {
    alpha: num(req.alpha, 1),
    l1Ratio: num(req.l1Ratio, 0.5),
    k: num(req.k, 5),
    maxDepth: num(req.maxDepth, 4),
    nEstimators: num(req.nEstimators, 50),
    rounds: num(req.rounds, 50),
    learningRate: num(req.learningRate, 0.1),
    hidden: num(req.hidden, 16),
    epochs: num(req.epochs, 200),
    lr: num(req.lr, 0.05),
    C: num(req.C, 1),
    l2: num(req.l2, 1),
    nClusters: num(req.nClusters, 3),
    radiusFactor: num(req.radiusFactor, 1.5),
  };
}

/** Deterministic RNG stream for model internals (bagging / MLP init /
 *  k-means). Slot 0 = the main run, slots 1..k = the CV folds — every
 *  training run gets its own stream so identical requests stay
 *  byte-identical. */
export function modelRng(effSeed: number, slot: number): () => number {
  return mulberry32((effSeed + 104729 + 6151 * slot) >>> 0);
}

/* ------------------------------------------------------------------ */
/* Regression fits                                                     */
/* ------------------------------------------------------------------ */

/** What every regression fit produces: a row predictor plus the honest
 *  parameter summary for the model family (coefficients for the linear
 *  family, normalized impurity-reduction importances for trees/ensembles,
 *  echoed hyperparameters for kNN/MLP/HSVR). */
export interface RegFit {
  predict(xsRaw: number[]): number;
  coefficients?: { intercept: number; coefs: number[]; standardized: boolean };
  featureImportance?: { features: string[]; importance: number[]; measure: "variance" | "gini" };
  modelParams?: Record<string, number>;
  /** HSVR only: per-cluster geometry + local-fit diagnostics. */
  hsvr?: {
    nClusters: number;
    radiusFactor: number;
    alpha: number;
    clusters: HsvrClusterInfo[];
  };
}

/** Fit any regression model over a raw (unstandardized) feature matrix. */
export function fitRegressionModel(
  model: string,
  XTrain: number[][],
  yTrain: number[],
  hp: Hyper,
  rand: () => number,
  featureNames: string[]
): RegFit {
  switch (model) {
    case "linear":
    case "polynomial":
    case "ridge": {
      /* Multivariate closed-form path (degree is 1 here; the single-feature
       * polynomial case is handled by the caller). */
      const effAlpha = model === "ridge" ? hp.alpha : 0;
      const m = fitMultivariateRegression(XTrain, yTrain, { alpha: effAlpha });
      return {
        predict: (xs) => m.predict(xs),
        coefficients: { intercept: m.w[0] ?? 0, coefs: m.w.slice(1), standardized: true },
      };
    }
    case "lasso":
    case "elastic_net": {
      const l1Ratio = model === "lasso" ? 1 : hp.l1Ratio;
      const m = fitElasticNet(XTrain, yTrain, { alpha: hp.alpha, l1Ratio });
      return {
        predict: (xs) => m.predict(xs),
        coefficients: { intercept: m.b, coefs: [...m.w], standardized: true },
        modelParams:
          model === "elastic_net"
            ? { alpha: hp.alpha, l1Ratio }
            : { alpha: hp.alpha },
      };
    }
    case "knn": {
      const m = fitKnnRegressor(XTrain, yTrain, { k: hp.k });
      return { predict: (xs) => m.predict(xs), modelParams: { k: hp.k } };
    }
    case "decision_tree": {
      const m = fitRegressionTree(XTrain, yTrain, { maxDepth: hp.maxDepth });
      return {
        predict: (xs) => m.predict(xs),
        featureImportance: {
          features: featureNames,
          importance: normalizeImportance(m.rawImportance),
          measure: "variance",
        },
      };
    }
    case "random_forest": {
      const m = fitRandomForestRegressor(XTrain, yTrain, {
        nEstimators: hp.nEstimators,
        maxDepth: hp.maxDepth,
        rand,
      });
      return {
        predict: (xs) => m.predict(xs),
        featureImportance: {
          features: featureNames,
          importance: m.importance,
          measure: "variance",
        },
      };
    }
    case "gradient_boosting": {
      const m = fitGradientBoostingRegressor(XTrain, yTrain, {
        rounds: hp.rounds,
        learningRate: hp.learningRate,
      });
      return {
        predict: (xs) => m.predict(xs),
        featureImportance: {
          features: featureNames,
          importance: m.importance,
          measure: "variance",
        },
      };
    }
    case "mlp": {
      const m = fitMlpRegressor(XTrain, yTrain, {
        hidden: hp.hidden,
        epochs: hp.epochs,
        lr: hp.lr,
        rand,
      });
      return {
        predict: (xs) => m.predict(xs),
        modelParams: { hidden: hp.hidden, epochs: hp.epochs, lr: hp.lr },
      };
    }
    case "hsvr": {
      const m = fitHsvr(XTrain, yTrain, {
        nClusters: hp.nClusters,
        radiusFactor: hp.radiusFactor,
        alpha: hp.alpha,
        rand,
      });
      return {
        predict: (xs) => m.predict(xs),
        modelParams: {
          nClusters: hp.nClusters,
          radiusFactor: hp.radiusFactor,
          alpha: hp.alpha,
        },
        hsvr: {
          nClusters: hp.nClusters,
          radiusFactor: hp.radiusFactor,
          alpha: hp.alpha,
          clusters: m.info,
        },
      };
    }
    default:
      throw new Error(`Unknown regression model "${model}".`);
  }
}

/* ------------------------------------------------------------------ */
/* Classification fits                                                 */
/* ------------------------------------------------------------------ */

/** What every classifier produces: a hard 0/1 prediction over raw feature
 *  pairs plus the honest parameter summary for the model family. */
export interface ClsFit {
  predict01(xRaw: number, yRaw: number): 0 | 1;
  coefficients?: { intercept: number; coefs: number[]; standardized: boolean };
  featureImportance?: { features: string[]; importance: number[]; measure: "variance" | "gini" };
  modelParams?: Record<string, number>;
}

/** Fit any classification model over raw (unstandardized) 2-feature rows. */
export function fitClassificationModel(
  model: string,
  XTrain: number[][],
  labelsTrain: number[],
  hp: Hyper,
  rand: () => number,
  featureNames: string[]
): ClsFit {
  switch (model) {
    case "logistic": {
      /* Legacy path: identical trainer invocation (batch GD, lr 0.1,
       * 800 epochs, L2 with bias exempt) for byte compatibility. */
      const pts = XTrain.map((row, i) => ({
        x: row[0] ?? 0,
        y: row[1] ?? 0,
        label: labelsTrain[i] ?? 0,
      }));
      const m = fitLogisticRegression(pts, { l2: hp.l2, lr: 0.1, epochs: 800 });
      return {
        predict01: (x, y) => (m.predict(x, y) >= 0.5 ? 1 : 0),
        coefficients: { intercept: m.bias, coefs: [...m.w], standardized: true },
      };
    }
    case "knn": {
      const m = fitKnnClassifier(XTrain, labelsTrain, { k: hp.k });
      return {
        predict01: (x, y) => m.predict([x, y]),
        modelParams: { k: hp.k },
      };
    }
    case "decision_tree": {
      const m = fitClassificationTree(XTrain, labelsTrain, { maxDepth: hp.maxDepth });
      return {
        predict01: (x, y) => m.predict([x, y]),
        featureImportance: {
          features: featureNames,
          importance: normalizeImportance(m.rawImportance),
          measure: "gini",
        },
      };
    }
    case "random_forest": {
      const m = fitRandomForestClassifier(XTrain, labelsTrain, {
        nEstimators: hp.nEstimators,
        maxDepth: hp.maxDepth,
        rand,
      });
      return {
        predict01: (x, y) => m.predict([x, y]),
        featureImportance: {
          features: featureNames,
          importance: m.importance,
          measure: "gini",
        },
      };
    }
    case "gaussian_nb": {
      const m = fitGaussianNB(XTrain, labelsTrain);
      return {
        predict01: (x, y) => m.predict([x, y]),
        modelParams: { varSmoothing: 1e-9 },
      };
    }
    case "svm_linear": {
      const m = fitLinearSvm(XTrain, labelsTrain, { C: hp.C });
      return {
        predict01: (x, y) => m.predict([x, y]),
        modelParams: { C: hp.C },
      };
    }
    case "mlp": {
      const m = fitMlpClassifier(XTrain, labelsTrain, {
        hidden: hp.hidden,
        epochs: hp.epochs,
        lr: hp.lr,
        rand,
      });
      return {
        predict01: (x, y) => m.predict([x, y]),
        modelParams: { hidden: hp.hidden, epochs: hp.epochs, lr: hp.lr },
      };
    }
    default:
      throw new Error(`Unknown classification model "${model}".`);
  }
}

/* ------------------------------------------------------------------ */
/* RandomSearch tuning spaces (keras-tuner-style Choice grids)         */
/* ------------------------------------------------------------------ */

/** One searchable hyperparameter slot with its discrete value grid. */
export interface TuneChoice {
  key: keyof HyperSource;
  values: number[];
}

/**
 * Per-model discrete search grids — the web mirror of the library's
 * `pharmaml.ml.tuner` RandomSearch spaces (Choice slots sampled by a seeded
 * stream). Models with no tunable hyperparameters (linear, polynomial,
 * gaussian_nb) are absent / empty and are rejected by the tune route.
 */
export const REG_TUNE_SPACES: Record<string, TuneChoice[]> = {
  ridge: [{ key: "alpha", values: [0.01, 0.1, 0.5, 1, 2, 5, 10] }],
  lasso: [{ key: "alpha", values: [0.01, 0.05, 0.1, 0.5, 1, 2] }],
  elastic_net: [
    { key: "alpha", values: [0.05, 0.1, 0.5, 1, 2] },
    { key: "l1Ratio", values: [0.1, 0.3, 0.5, 0.7, 0.9] },
  ],
  knn: [{ key: "k", values: [1, 3, 5, 7, 9, 11, 15, 21] }],
  decision_tree: [{ key: "maxDepth", values: [2, 3, 4, 5, 6, 7, 8, 10] }],
  random_forest: [
    { key: "nEstimators", values: [10, 25, 50, 100, 150] },
    { key: "maxDepth", values: [3, 4, 6, 8, 10] },
  ],
  gradient_boosting: [
    { key: "rounds", values: [20, 50, 100, 150] },
    { key: "learningRate", values: [0.02, 0.05, 0.1, 0.2, 0.3] },
  ],
  mlp: [
    { key: "hidden", values: [4, 8, 16, 32, 48] },
    { key: "epochs", values: [100, 200, 400, 800] },
    { key: "lr", values: [0.01, 0.02, 0.05, 0.1, 0.2] },
  ],
  hsvr: [
    { key: "nClusters", values: [2, 3, 4, 5, 6, 8] },
    { key: "radiusFactor", values: [0.5, 1, 1.5, 2, 3] },
    { key: "alpha", values: [0.01, 0.1, 1, 10] },
  ],
};

export const CLS_TUNE_SPACES: Record<string, TuneChoice[]> = {
  logistic: [{ key: "l2", values: [0.01, 0.1, 0.5, 1, 2, 5, 10] }],
  knn: [{ key: "k", values: [1, 3, 5, 7, 9, 11, 15, 21] }],
  decision_tree: [{ key: "maxDepth", values: [2, 3, 4, 5, 6, 7, 8, 10] }],
  random_forest: [
    { key: "nEstimators", values: [10, 25, 50, 100] },
    { key: "maxDepth", values: [3, 4, 6, 8, 10] },
  ],
  svm_linear: [{ key: "C", values: [0.05, 0.1, 0.5, 1, 2, 5, 10] }],
  mlp: [
    { key: "hidden", values: [4, 8, 16, 32] },
    { key: "epochs", values: [200, 400, 800] },
    { key: "lr", values: [0.02, 0.05, 0.1, 0.2] },
  ],
};

/** Total configurations in a space (product of the per-slot grids). */
export function tuneSpaceSize(choices: TuneChoice[]): number {
  return choices.reduce((acc, c) => acc * Math.max(c.values.length, 1), 1);
}

/**
 * Decode a flat index in [0, spaceSize) into a parameter assignment
 * (mixed-radix digits — the deterministic inverse of the grid layout).
 */
export function decodeTuneIndex(
  choices: TuneChoice[],
  index: number
): Record<string, number> {
  const params: Record<string, number> = {};
  let rest = index;
  for (let i = choices.length - 1; i >= 0; i--) {
    const c = choices[i]!;
    const radix = Math.max(c.values.length, 1);
    const digit = rest % radix;
    rest = Math.floor(rest / radix);
    params[c.key] = c.values[digit] ?? c.values[c.values.length - 1] ?? 0;
  }
  return params;
}

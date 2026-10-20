/**
 * Shared ML model registry — the single source of truth for the
 * `/api/ml/train` and `/api/ml/tune` routes (and the UI's mirrored model
 * list). Fitting logic, hyperparameter resolution, and the RandomSearch
 * tuning grids all live here so the two routes can never drift apart.
 */

import {
  fitAdaBoostClassifier,
  fitAdaBoostRegressor,
  fitClassificationTree,
  fitElasticNet,
  fitExtraTreesClassifier,
  fitExtraTreesRegressor,
  fitGaussianNB,
  fitGradientBoostingRegressor,
  fitHuberRegression,
  fitHsvr,
  fitKernelRidgeRbf,
  fitKnnClassifier,
  fitKnnRegressor,
  fitLinearSvm,
  fitLogisticRegression,
  fitMlpClassifier,
  fitMlpRegressor,
  fitMultivariateRegression,
  fitRandomForestClassifier,
  fitRandomForestRegressor,
  fitRbfSvmClassifier,
  fitRegressionTree,
  mulberry32,
  normalizeImportance,
  type HsvrClusterInfo,
} from "@/lib/ml-math";

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

/** Regression models: 3 legacy (byte-compatible) + 12 (incl. hsvr, huber,
 *  ada_boost, svm_rbf, extra_trees — the web mirror of the library's
 *  adaboost / huber_regression / svm / extra_trees families). */
export const REGRESSION_MODELS = [
  "linear",
  "polynomial",
  "ridge",
  "lasso",
  "elastic_net",
  "huber",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "gradient_boosting",
  "ada_boost",
  "svm_rbf",
  "mlp",
  "hsvr",
] as const;

/** Classification models: legacy logistic + 10 (incl. ada_boost,
 *  extra_trees and the RBF-kernel SVM — the library's adaboost /
 *  extra_trees / svm families). */
export const CLASSIFICATION_MODELS = [
  "logistic",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "ada_boost",
  "gaussian_nb",
  "svm_linear",
  "svm_rbf",
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
  /** Huber: quadratic-core / linear-tail switch point (target units). */
  epsilon?: number;
  /** RBF-kernel models: the Gaussian kernel width γ. */
  gamma?: number;
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
  epsilon: number;
  gamma: number;
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
    epsilon: num(req.epsilon, 1.35),
    gamma: num(req.gamma, 0.5),
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
  /** HSVR only: per-cluster geometry + local-fit diagnostics + the
   *  train-row scatter geometry for the trust-radius plot. */
  hsvr?: {
    nClusters: number;
    radiusFactor: number;
    alpha: number;
    clusters: HsvrClusterInfo[];
    /** The trust-radius scatter (train rows + centroids + axis labels). */
    scatter: {
      dimA: string;
      dimB: string;
      bIsTarget: boolean;
      projected: boolean;
      points: Array<{ a: number; b: number; cluster: number }>;
      centroids: Array<{ a: number; b: number; radius: number }>;
    };
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
    case "huber": {
      /* Robust IRLS — the library's huber_regression family. The fit
       * surfaces how many train rows ended up in the quadratic core vs
       * the linear tail (the honest robustness diagnostic). */
      const m = fitHuberRegression(XTrain, yTrain, { epsilon: hp.epsilon });
      return {
        predict: (xs) => m.predict(xs),
        coefficients: { intercept: m.b, coefs: [...m.w], standardized: true },
        modelParams: {
          epsilon: hp.epsilon,
          nRobust: m.nRobust,
          nOutlier: m.nOutlier,
        },
      };
    }
    case "ada_boost": {
      const m = fitAdaBoostRegressor(XTrain, yTrain, {
        nEstimators: hp.nEstimators,
        learningRate: hp.learningRate,
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
    case "svm_rbf": {
      const m = fitKernelRidgeRbf(XTrain, yTrain, {
        gamma: hp.gamma,
        alpha: hp.alpha,
      });
      return {
        predict: (xs) => m.predict(xs),
        modelParams: { gamma: hp.gamma, lambda: hp.alpha },
      };
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
    case "extra_trees": {
      const m = fitExtraTreesRegressor(XTrain, yTrain, {
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
      /* Axis labels for the trust-radius scatter: the first standardized
       *  feature vs the target (single feature) or the second standardized
       *  feature (a projection when the model has more than two). */
      const singleFeature = (XTrain[0]?.length ?? 0) <= 1;
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
          scatter: {
            dimA: `${featureNames[0] ?? "feature"} (standardized)`,
            dimB: singleFeature
              ? "target"
              : `${featureNames[1] ?? "feature"} (standardized)`,
            bIsTarget: singleFeature,
            projected: (XTrain[0]?.length ?? 0) > 2,
            points: m.scatter.points,
            centroids: m.scatter.centroids,
          },
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
    case "extra_trees": {
      const m = fitExtraTreesClassifier(XTrain, labelsTrain, {
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
    case "ada_boost": {
      const m = fitAdaBoostClassifier(XTrain, labelsTrain, {
        nEstimators: hp.nEstimators,
        learningRate: hp.learningRate,
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
    case "svm_rbf": {
      const m = fitRbfSvmClassifier(XTrain, labelsTrain, {
        C: hp.C,
        gamma: hp.gamma,
      });
      return {
        predict01: (x, y) => m.predict([x, y]),
        modelParams: { C: hp.C, gamma: hp.gamma, lambda: m.lambda, nSupport: m.nSupport },
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
  huber: [{ key: "epsilon", values: [1.05, 1.35, 1.75, 2.2, 3] }],
  knn: [{ key: "k", values: [1, 3, 5, 7, 9, 11, 15, 21] }],
  decision_tree: [{ key: "maxDepth", values: [2, 3, 4, 5, 6, 7, 8, 10] }],
  random_forest: [
    { key: "nEstimators", values: [10, 25, 50, 100, 150] },
    { key: "maxDepth", values: [3, 4, 6, 8, 10] },
  ],
  extra_trees: [
    { key: "nEstimators", values: [10, 25, 50, 100, 150] },
    { key: "maxDepth", values: [3, 4, 6, 8, 10] },
  ],
  gradient_boosting: [
    { key: "rounds", values: [20, 50, 100, 150] },
    { key: "learningRate", values: [0.02, 0.05, 0.1, 0.2, 0.3] },
  ],
  ada_boost: [
    { key: "nEstimators", values: [10, 25, 50, 100, 150] },
    { key: "learningRate", values: [0.05, 0.1, 0.2, 0.3, 0.5] },
  ],
  svm_rbf: [
    { key: "gamma", values: [0.05, 0.1, 0.25, 0.5, 1, 2] },
    { key: "alpha", values: [0.01, 0.1, 1, 10] },
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
  extra_trees: [
    { key: "nEstimators", values: [10, 25, 50, 100] },
    { key: "maxDepth", values: [3, 4, 6, 8, 10] },
  ],
  ada_boost: [
    { key: "nEstimators", values: [10, 25, 50, 100] },
    { key: "learningRate", values: [0.05, 0.1, 0.2, 0.3, 0.5] },
  ],
  svm_linear: [{ key: "C", values: [0.05, 0.1, 0.5, 1, 2, 5, 10] }],
  svm_rbf: [
    { key: "gamma", values: [0.05, 0.1, 0.25, 0.5, 1, 2] },
    { key: "C", values: [0.05, 0.1, 0.5, 1, 2, 5, 10] },
  ],
  mlp: [
    { key: "hidden", values: [4, 8, 16, 32] },
    { key: "epochs", values: [200, 400, 800] },
    { key: "lr", values: [0.02, 0.05, 0.1, 0.2] },
  ],
};

/* ------------------------------------------------------------------ */
/* Hyperband resources                                                 */
/* ------------------------------------------------------------------ */

/**
 * Models whose search space contains a resource hyperparameter — a slot
 * whose value scales compute monotonically, which successive halving can
 * promote rung by rung (the web mirror of `pharmaml.ml.tuner.Hyperband`'s
 * resource contract). Other models have no resource and are rejected by
 * the tune route's Hyperband mode with an actionable message.
 */
export const TUNE_RESOURCES: Record<string, { key: string; label: string }> = {
  mlp: { key: "epochs", label: "epochs" },
  random_forest: { key: "nEstimators", label: "estimators" },
  extra_trees: { key: "nEstimators", label: "estimators" },
  gradient_boosting: { key: "rounds", label: "boosting rounds" },
  ada_boost: { key: "nEstimators", label: "boosting rounds" },
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

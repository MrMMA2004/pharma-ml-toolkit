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
  fitGaussianProcess,
  fitGaussianProcessClassification,
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
  fitMultivariateLogistic,
  fitMultivariateRegression,
  fitPlsRegression,
  fitRandomForestClassifier,
  fitRandomForestRegressor,
  fitRbfSvmClassifier,
  fitRegressionTree,
  kFoldSplit,
  mulberry32,
  normalizeImportance,
  regressionMetrics,
  type HsvrClusterInfo,
} from "@/lib/ml-math";

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

/** Regression models: 3 legacy (byte-compatible) + 14 (incl. hsvr, huber,
 *  ada_boost, svm_rbf, extra_trees, pls, gaussian_process — the web
 *  mirror of the library's adaboost / huber_regression / svm /
 *  extra_trees / pls / gaussian_process families) + the voting and
 *  stacking meta-ensembles (RF + SVM + KNN level-0 — the mirror of the
 *  library's `build_voting` / `build_stacking` registrations). */
export const REGRESSION_MODELS = [
  "linear",
  "polynomial",
  "ridge",
  "lasso",
  "elastic_net",
  "huber",
  "pls",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "gradient_boosting",
  "ada_boost",
  "svm_rbf",
  "gaussian_process",
  "mlp",
  "hsvr",
  "voting",
  "stacking",
] as const;

/** Classification models: legacy logistic + 10 (incl. ada_boost,
 *  extra_trees, the RBF-kernel SVM and gaussian_process — the library's
 *  adaboost / extra_trees / svm / gaussian_process_classifier
 *  families; the web id is task-scoped like svm_rbf) + the voting and
 *  stacking meta-ensembles (soft voting / OOF stacking over the same
 *  RF + SVM + KNN trio). */
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
  "gaussian_process",
  "mlp",
  "voting",
  "stacking",
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
  /** PLS: latent component count (rank-capped at extraction). */
  nComponents?: number;
  /** GP: RBF length scale ℓ (standardized-feature units). */
  lengthScale?: number;
  /** GP: signal amplitude σf (raw target units). */
  signalStd?: number;
  /** GP: white-noise observation std σn (raw target units). */
  noiseStd?: number;
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
  nComponents: number;
  lengthScale: number;
  signalStd: number;
  noiseStd: number;
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
    nComponents: num(req.nComponents, 2),
    lengthScale: num(req.lengthScale, 1),
    signalStd: num(req.signalStd, 1),
    noiseStd: num(req.noiseStd, 0.1),
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
/* Meta-ensembles (voting / stacking)                                  */
/* ------------------------------------------------------------------ */

/** The level-0 trio the library's `_ENSEMBLE_BASE_MODELS` registers —
 *  RF + SVM + KNN, the same members the rar's own Voting.py /
 *  Stacking.py stacked. Member hyperparameters are read from the flat
 *  `Hyper` slots each family already owns (nEstimators/maxDepth → RF,
 *  gamma/α-or-C → SVM, k → KNN). */
export const ENSEMBLE_MEMBERS = ["random_forest", "svm_rbf", "knn"] as const;

/** Equal member weights — the library's VotingClassifier default
 *  (weights=None) and the (1, 1, 1) first grid entry. */
export const ENSEMBLE_WEIGHTS = [1, 1, 1] as const;

/** Out-of-fold count for the stacking meta-learner's training matrix —
 *  the library's `build_stacking` pins cv=5. */
export const ENSEMBLE_OOF_FOLDS = 5;

/** Serializable ensemble readout — what the train route echoes so the
 *  UI can answer "did the ensemble actually help?" */
export interface EnsembleReadout {
  kind: "voting" | "stacking";
  /** How members combine: mean (regression both kinds) or averaged
   *  probability (classification, sklearn's soft voting). */
  rule: "average" | "soft";
  /** Registry ids of the level-0 members, blend order. */
  members: string[];
  /** Blend weight per member (equal by default). */
  weights: number[];
  /** Stacking only: the level-1 learner (ridge for regression,
   *  logistic for classification — the library defaults). */
  metaModel?: "ridge" | "logistic";
  /** Stacking only: folds that built the OOF matrix. */
  oofFolds?: number;
  /** Stacking only: cross-validated score per level-0 member on the
   *  OOF predictions the meta-learner was trained against (R² for
   *  regression, accuracy for classification). */
  oof?: { name: string; score: number }[];
  /** Stacking only: the meta-learner's weights on the (standardized)
   *  OOF columns — which base model the blender actually trusts. */
  metaCoefs?: { intercept: number; coefs: number[] };
}

/** ------------------------------------------------------------------
 * Ensemble internals: deterministic per-member RNG streams derived
 * from the fit's own stream (drawn once, then split per member so
 * bagging seeds stay stable when members are added). */
function memberRngs(rand: () => number, count: number): Array<() => number> {
  const seed = Math.floor(rand() * 2 ** 31) >>> 0;
  return Array.from({ length: count }, (_, i) => mulberry32((seed + 7919 * (i + 1)) >>> 0));
}

/** Deterministic OOF fold split for stacking (independent of the route's
 *  CV shuffle — drawn from the same per-fit stream). */
function ensembleFoldRand(rand: () => number): () => number {
  const seed = Math.floor(rand() * 2 ** 31) >>> 0;
  return mulberry32((seed + 104729) >>> 0);
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
  /** GP only: predictive standard deviation in raw target units —
   *  the ±1.96σ band the chart draws. */
  predictStd?: (xsRaw: number[]) => number;
  /** PLS only: VIP scores (VIP > 1 marks an influential descriptor —
   *  the classic QSAR readout the coefficients card never showed). */
  vip?: { features: string[]; values: number[] };
  /** GP only: the kernel readout — hyperparameters (learned by the
   *  deterministic LML grid search when the request left them unset)
   *  plus the log marginal likelihood of the train targets under them. */
  gp?: {
    lengthScale: number;
    signalStd: number;
    noiseStd: number;
    logMarginalLikelihood: number;
    nSupport: number;
    kernelTuned: boolean;
  };
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
  /** Voting / stacking only: the meta-model readout (members, rule,
   *  OOF scores, meta weights — everything except the predictors). */
  ensemble?: EnsembleReadout;
  /** Voting / stacking only: each level-0 member's own predictor so
   *  the route can score every member on the held-out rows next to
   *  the ensemble itself (the "did it help" comparison). */
  memberPredict?: { name: string; predict(xsRaw: number[]): number }[];
}

/** Extra fitting directives shared by the train route's main fit and its
 *  CV folds (the tune/compare routes keep the default fixed-kernel
 *  behaviour — their trials always carry explicit hyperparameters). */
export interface FitDirectives {
  /** GP: learn the kernel (lengthScale/signalStd/noiseStd) by the
   *  deterministic marginal-likelihood grid search instead of using
   *  the request's (default-filled) values — the web mirror of the
   *  library's LML-tuned kernel. */
  learnGpKernel?: boolean;
  /** GP classification: learn (lengthScale/signalStd) by the Laplace-
   *  lml grid search (the classification twin of learnGpKernel — no
   *  noise slot: the logistic likelihood absorbs what a WhiteKernel
   *  would model). */
  learnGpClsKernel?: boolean;
}

/** Fit any regression model over a raw (unstandardized) feature matrix. */
export function fitRegressionModel(
  model: string,
  XTrain: number[][],
  yTrain: number[],
  hp: Hyper,
  rand: () => number,
  featureNames: string[],
  directives: FitDirectives = {}
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
    case "pls": {
      /* NIPALS PLS1 — the library's `pls` family. The fit surfaces the
       * component count actually extracted (rank-limited) alongside the
       * standardized-space weights; PLS is linear once the latent space
       * is chosen, so the coefficients card is the honest summary. The
       * VIP scores ride along too — VIP > 1 marks an influential
       * descriptor (the classic QSAR readout, previously computed and
       * dropped). */
      const m = fitPlsRegression(XTrain, yTrain, { nComponents: hp.nComponents });
      return {
        predict: (xs) => m.predict(xs),
        coefficients: { intercept: m.intercept, coefs: [...m.coefs], standardized: true },
        modelParams: { nComponents: m.nUsed },
        vip: { features: featureNames, values: [...m.vip] },
      };
    }
    case "gaussian_process": {
      /* Kriging — the library's `gaussian_process` family. With the learn
       * directive (the request left the kernel hyperparameters unset) the
       * deterministic LML grid search picks them — the web mirror of
       * sklearn's marginal-likelihood-tuned kernel; explicit values fix
       * the kernel. Predictive mean + per-point σ via the Cholesky solve. */
      const m = fitGaussianProcess(
        XTrain,
        yTrain,
        directives.learnGpKernel
          ? {}
          : { lengthScale: hp.lengthScale, signalStd: hp.signalStd, noiseStd: hp.noiseStd }
      );
      return {
        predict: (xs) => m.predict(xs),
        predictStd: (xs) => m.predictStd(xs),
        modelParams: {
          lengthScale: m.lengthScale,
          signalStd: m.signalStd,
          noiseStd: m.noiseStd,
        },
        gp: {
          lengthScale: m.lengthScale,
          signalStd: m.signalStd,
          noiseStd: m.noiseStd,
          logMarginalLikelihood: m.logMarginalLikelihood,
          nSupport: m.nSupport,
          kernelTuned: m.kernelTuned,
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
    case "voting":
    case "stacking":
      /* Meta-ensembles: RF + SVM + KNN level-0, blended by the weighted
       * mean (voting) or a ridge meta-learner over the 5-fold OOF
       * matrix (stacking) — the web mirror of the library's
       * build_voting / build_stacking registrations. */
      return fitRegressionEnsemble(model, XTrain, yTrain, hp, rand);
    default:
      throw new Error(`Unknown regression model "${model}".`);
  }
}

/* ------------------------------------------------------------------ */
/* Ensemble member fits (regression)                                   */
/* ------------------------------------------------------------------ */

/**
 * Fit the level-0 trio over raw rows — RF (bagged, seeded), RBF kernel
 * ridge (the svm_rbf family), and kNN. Each member reads only its own
 * `Hyper` slots, so one flat hyperparameter request configures the whole
 * stack exactly like the library's `build_voting` / `build_stacking`
 * over their default member factories.
 */
function fitRegressionMembers(
  XTrain: number[][],
  yTrain: number[],
  hp: Hyper,
  rand: () => number
): { name: string; predict(xsRaw: number[]): number }[] {
  const rngs = memberRngs(rand, ENSEMBLE_MEMBERS.length);
  const rf = fitRandomForestRegressor(XTrain, yTrain, {
    nEstimators: hp.nEstimators,
    maxDepth: hp.maxDepth,
    rand: rngs[0]!,
  });
  const svm = fitKernelRidgeRbf(XTrain, yTrain, { gamma: hp.gamma, alpha: hp.alpha });
  const knn = fitKnnRegressor(XTrain, yTrain, { k: hp.k });
  return [
    { name: ENSEMBLE_MEMBERS[0]!, predict: (xs) => rf.predict(xs) },
    { name: ENSEMBLE_MEMBERS[1]!, predict: (xs) => svm.predict(xs) },
    { name: ENSEMBLE_MEMBERS[2]!, predict: (xs) => knn.predict(xs) },
  ];
}

/** Weighted mean of the member predictions (VotingRegressor's rule). */
function blendPredictions(
  members: { name: string; predict(xsRaw: number[]): number }[],
  weights: number[]
): (xsRaw: number[]) => number {
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  return (xs) => {
    let acc = 0;
    for (let i = 0; i < members.length; i++) {
      acc += (weights[i] ?? 1) * (members[i]?.predict(xs) ?? 0);
    }
    return acc / wSum;
  };
}

/** Fit the voting / stacking regression ensembles over raw rows. */
function fitRegressionEnsemble(
  kind: "voting" | "stacking",
  XTrain: number[][],
  yTrain: number[],
  hp: Hyper,
  rand: () => number
): RegFit {
  const weights = [...ENSEMBLE_WEIGHTS];
  if (kind === "voting") {
    /* VotingRegressor: the weighted mean of the level-0 predictions —
     * the voting rule is ignored for regression in the library too. */
    const members = fitRegressionMembers(XTrain, yTrain, hp, rand);
    return {
      predict: blendPredictions(members, weights),
      modelParams: { nMembers: members.length },
      ensemble: {
        kind: "voting",
        rule: "average",
        members: [...ENSEMBLE_MEMBERS],
        weights,
      },
      memberPredict: members.map((m) => ({ name: m.name, predict: m.predict })),
    };
  }

  /* StackingRegressor: build the level-0 out-of-fold matrix (cv = 5,
   * the library's build_stacking pin), fit the ridge meta-learner on
   * it, then refit every member on the full train rows. Prediction =
   * meta over the members' fresh predictions — sklearn's stack
   * contract, mirrored. */
  const n = XTrain.length;
  const foldRand = ensembleFoldRand(rand);
  const folds = kFoldSplit(n, ENSEMBLE_OOF_FOLDS, foldRand);
  const oof: number[][] = XTrain.map(() => new Array<number>(ENSEMBLE_MEMBERS.length).fill(0));
  for (const fold of folds) {
    const foldSet = new Set(fold);
    const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
    const Xf = trainIdx.map((i) => XTrain[i] ?? []);
    const yf = trainIdx.map((i) => yTrain[i] ?? 0);
    /* Per-fold member fits: fresh deterministic streams derived from
     * the fold's own index arithmetic (no reuse of the main stream —
     * the OOF machinery must not perturb the full-train seeds). */
    const foldSeed = (Math.floor(rand() * 2 ** 31) + fold.length * 31) >>> 0;
    const rf = fitRandomForestRegressor(Xf, yf, {
      nEstimators: hp.nEstimators,
      maxDepth: hp.maxDepth,
      rand: mulberry32(foldSeed),
    });
    const svm = fitKernelRidgeRbf(Xf, yf, { gamma: hp.gamma, alpha: hp.alpha });
    const knn = fitKnnRegressor(Xf, yf, { k: hp.k });
    const foldMembers = [rf, svm, knn];
    for (const i of fold) {
      for (let c = 0; c < foldMembers.length; c++) {
        oof[i]![c] = foldMembers[c]!.predict(XTrain[i] ?? []);
      }
    }
  }
  /* The honest level-0 readout: each member's OOF R² on the exact
   * matrix the meta-learner trains against. */
  const oofScores = ENSEMBLE_MEMBERS.map((name, c) => ({
    name,
    score: regressionMetrics(yTrain, oof.map((row) => row[c] ?? 0)).r2,
  }));

  const meta = fitMultivariateRegression(oof, yTrain, { alpha: 1 });
  const members = fitRegressionMembers(XTrain, yTrain, hp, rand);
  return {
    predict: (xs) => meta.predict(members.map((m) => m.predict(xs))),
    modelParams: { nMembers: members.length, oofFolds: ENSEMBLE_OOF_FOLDS },
    ensemble: {
      kind: "stacking",
      rule: "average",
      members: [...ENSEMBLE_MEMBERS],
      weights,
      metaModel: "ridge",
      oofFolds: ENSEMBLE_OOF_FOLDS,
      oof: oofScores,
      metaCoefs: { intercept: meta.w[0] ?? 0, coefs: meta.w.slice(1) },
    },
    memberPredict: members.map((m) => ({ name: m.name, predict: m.predict })),
  };
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
  /** GP classification: calibrated class-1 probability at a raw row. */
  predictProba?: (xRaw: number, yRaw: number) => number;
  /** GP classification only: the Laplace posterior readout. */
  gpCls?: {
    lengthScale: number;
    signalStd: number;
    newtonIterations: number;
    logMarginalLikelihood: number;
    kernelTuned: boolean;
  };
  /** Voting / stacking only: the meta-model readout (members, rule,
   *  OOF scores, meta weights — everything except the predictors). */
  ensemble?: EnsembleReadout;
  /** Voting / stacking only: each level-0 member's own hard classifier
   *  so the route can score every member on the held-out rows next to
   *  the ensemble itself. */
  memberPredict01?: { name: string; predict01(xRaw: number, yRaw: number): 0 | 1 }[];
}

/** Fit any classification model over raw (unstandardized) 2-feature rows. */
export function fitClassificationModel(
  model: string,
  XTrain: number[][],
  labelsTrain: number[],
  hp: Hyper,
  rand: () => number,
  featureNames: string[],
  directives: FitDirectives = {}
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
    case "gaussian_process": {
      /* The classification half of the GP family — the web mirror of
       *  the library's gaussian_process_classifier (Laplace posterior:
       *  Newton mode + 5-erf predictive probabilities, verified against
       *  sklearn by scratch/verify_gpc_parity.{py,ts}). With the learn
       *  directive (request left ℓ / σf unset) the Laplace-lml grid
       *  search picks the kernel. */
      const m = fitGaussianProcessClassification(
        XTrain,
        labelsTrain,
        directives.learnGpClsKernel ? {} : { lengthScale: hp.lengthScale, signalStd: hp.signalStd }
      );
      return {
        predict01: (x, y) => m.predict01([x, y]),
        predictProba: (x, y) => m.predictProba([x, y]),
        modelParams: {
          lengthScale: m.lengthScale,
          signalStd: m.signalStd,
          newtonIterations: m.newtonIterations,
        },
        gpCls: {
          lengthScale: m.lengthScale,
          signalStd: m.signalStd,
          newtonIterations: m.newtonIterations,
          logMarginalLikelihood: m.logMarginalLikelihood,
          kernelTuned: m.kernelTuned,
        },
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
    case "voting":
    case "stacking":
      /* Meta-ensembles: soft vote (voting) or the OOF-stacked logistic
       * meta-learner (stacking) over the RF + SVM + KNN trio — the
       * mirror of the library's build_voting(voting="soft") /
       * build_stacking classification registrations. */
      return fitClassificationEnsemble(model, XTrain, labelsTrain, hp, rand);
    default:
      throw new Error(`Unknown classification model "${model}".`);
  }
}

/* ------------------------------------------------------------------ */
/* Ensemble member fits (classification)                               */
/* ------------------------------------------------------------------ */

/** A level-0 classification member: hard vote + class-1 probability
 *  over raw (x, y) feature pairs. Every member of the trio exposes a
 *  probability (RF leaf fractions, LS-SVM logistic decision read, kNN
 *  neighbour fractions), so sklearn's soft voting is available — the
 *  library's classification default. */
interface ClsMember {
  name: string;
  proba(xRaw: number, yRaw: number): number;
  predict01(xRaw: number, yRaw: number): 0 | 1;
}

function fitClassificationMembers(
  XTrain: number[][],
  labelsTrain: number[],
  hp: Hyper,
  rand: () => number
): ClsMember[] {
  const rngs = memberRngs(rand, ENSEMBLE_MEMBERS.length);
  const rf = fitRandomForestClassifier(XTrain, labelsTrain, {
    nEstimators: hp.nEstimators,
    maxDepth: hp.maxDepth,
    rand: rngs[0]!,
  });
  const svm = fitRbfSvmClassifier(XTrain, labelsTrain, { C: hp.C, gamma: hp.gamma });
  const knn = fitKnnClassifier(XTrain, labelsTrain, { k: hp.k });
  return [
    {
      name: ENSEMBLE_MEMBERS[0]!,
      proba: (x, y) => rf.predictProba?.([x, y]) ?? (rf.predict([x, y]) as number),
      predict01: (x, y) => rf.predict([x, y]),
    },
    {
      name: ENSEMBLE_MEMBERS[1]!,
      proba: (x, y) => svm.predictProba?.([x, y]) ?? (svm.predict([x, y]) as number),
      predict01: (x, y) => svm.predict([x, y]),
    },
    {
      name: ENSEMBLE_MEMBERS[2]!,
      proba: (x, y) => knn.predictProba?.([x, y]) ?? (knn.predict([x, y]) as number),
      predict01: (x, y) => knn.predict([x, y]),
    },
  ];
}

/** Fit the voting / stacking classification ensembles over raw rows. */
function fitClassificationEnsemble(
  kind: "voting" | "stacking",
  XTrain: number[][],
  labelsTrain: number[],
  hp: Hyper,
  rand: () => number
): ClsFit {
  const weights = [...ENSEMBLE_WEIGHTS];
  if (kind === "voting") {
    /* VotingClassifier with voting="soft": the weighted mean of the
     * members' class-1 probabilities (the library's classification
     * default). The averaged probability rides along as predictProba —
     * the boundary fade and the honest probabilistic scores come free. */
    const members = fitClassificationMembers(XTrain, labelsTrain, hp, rand);
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;
    const proba = (x: number, y: number): number => {
      let acc = 0;
      for (let i = 0; i < members.length; i++) {
        acc += (weights[i] ?? 1) * (members[i]?.proba(x, y) ?? 0.5);
      }
      return acc / wSum;
    };
    return {
      predict01: (x, y) => (proba(x, y) >= 0.5 ? 1 : 0),
      predictProba: proba,
      modelParams: { nMembers: members.length },
      ensemble: {
        kind: "voting",
        rule: "soft",
        members: [...ENSEMBLE_MEMBERS],
        weights,
      },
      memberPredict01: members.map((m) => ({ name: m.name, predict01: m.predict01 })),
    };
  }

  /* StackingClassifier: the level-0 OOF probability matrix (cv = 5)
   * trains the multivariate logistic meta-learner; members are then
   * refit on the full train rows and the meta predicts from their
   * fresh probabilities — sklearn's stack contract, mirrored. */
  const n = XTrain.length;
  const foldRand = ensembleFoldRand(rand);
  const folds = kFoldSplit(n, ENSEMBLE_OOF_FOLDS, foldRand);
  const oof: number[][] = XTrain.map(() => new Array<number>(ENSEMBLE_MEMBERS.length).fill(0.5));
  for (const fold of folds) {
    const foldSet = new Set(fold);
    const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
    const Xf = trainIdx.map((i) => XTrain[i] ?? []);
    const lf = trainIdx.map((i) => labelsTrain[i] ?? 0);
    const foldSeed = (Math.floor(rand() * 2 ** 31) + fold.length * 31) >>> 0;
    const rf = fitRandomForestClassifier(Xf, lf, {
      nEstimators: hp.nEstimators,
      maxDepth: hp.maxDepth,
      rand: mulberry32(foldSeed),
    });
    const svm = fitRbfSvmClassifier(Xf, lf, { C: hp.C, gamma: hp.gamma });
    const knn = fitKnnClassifier(Xf, lf, { k: hp.k });
    const foldMembers = [rf, svm, knn];
    for (const i of fold) {
      const row = XTrain[i] ?? [];
      for (let c = 0; c < foldMembers.length; c++) {
        oof[i]![c] = foldMembers[c]!.predictProba?.(row) ?? 0.5;
      }
    }
  }
  /* The honest level-0 readout: each member's OOF accuracy (hard vote
   * of its OOF probabilities) on the exact matrix the meta trains on. */
  const oofScores = ENSEMBLE_MEMBERS.map((name, c) => {
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const pred = (oof[i]?.[c] ?? 0.5) >= 0.5 ? 1 : 0;
      if (pred === (labelsTrain[i] ?? 0)) correct++;
    }
    return { name, score: correct / Math.max(n, 1) };
  });

  const meta = fitMultivariateLogistic(oof, labelsTrain, { l2: 1, lr: 0.1, epochs: 800 });
  const members = fitClassificationMembers(XTrain, labelsTrain, hp, rand);
  const proba = (x: number, y: number): number =>
    meta.predictProba(members.map((m) => m.proba(x, y)));
  return {
    predict01: (x, y) => (proba(x, y) >= 0.5 ? 1 : 0),
    predictProba: proba,
    modelParams: { nMembers: members.length, oofFolds: ENSEMBLE_OOF_FOLDS },
    ensemble: {
      kind: "stacking",
      rule: "soft",
      members: [...ENSEMBLE_MEMBERS],
      weights,
      metaModel: "logistic",
      oofFolds: ENSEMBLE_OOF_FOLDS,
      oof: oofScores,
      metaCoefs: { intercept: meta.bias, coefs: [...meta.w] },
    },
    memberPredict01: members.map((m) => ({ name: m.name, predict01: m.predict01 })),
  };
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
  pls: [{ key: "nComponents", values: [1, 2, 3, 4, 5, 6] }],
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
  gaussian_process: [
    { key: "lengthScale", values: [0.25, 0.5, 1, 2, 4] },
    { key: "signalStd", values: [0.5, 1, 2] },
    { key: "noiseStd", values: [0.01, 0.05, 0.2, 0.5] },
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
  gaussian_process: [
    { key: "lengthScale", values: [0.25, 0.5, 1, 2, 4] },
    { key: "signalStd", values: [0.5, 1, 2] },
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

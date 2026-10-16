import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  hashCsvSeed,
  parseCsv,
  resolveClassification,
  resolveMultiRegression,
  resolveRegression,
  zscore,
  type CsvClassificationData,
  type CsvMultiRegressionData,
  type CsvRegressionData,
} from "@/lib/csv";
import {
  classificationMetrics,
  fitClassificationTree,
  fitElasticNet,
  fitGaussianNB,
  fitGradientBoostingRegressor,
  fitKnnClassifier,
  fitKnnRegressor,
  fitLinearRegression,
  fitLinearSvm,
  fitLogisticRegression,
  fitMlpClassifier,
  fitMlpRegressor,
  fitMultivariateRegression,
  fitRandomForestClassifier,
  fitRandomForestRegressor,
  fitRegressionTree,
  kFoldSplit,
  makeMoons,
  makeRegressionData,
  mean,
  mulberry32,
  normalizeImportance,
  populationStd,
  regressionMetrics,
  shuffledSplit,
} from "@/lib/ml-math";
import type { CVFold, CsvSourceMeta, MLTrainResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const cvSchema = z.union([z.literal(0), z.literal(3), z.literal(5), z.literal(10)]);

/* ------------------------------------------------------------------ */
/* Model registry                                                      */
/* ------------------------------------------------------------------ */

/** Regression models: 3 legacy (byte-compatible) + 7 rebuilt. */
const REGRESSION_MODELS = [
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
] as const;

/** Classification models: legacy logistic + 6 rebuilt. */
const CLASSIFICATION_MODELS = [
  "logistic",
  "knn",
  "decision_tree",
  "random_forest",
  "gaussian_nb",
  "svm_linear",
  "mlp",
] as const;

/** Hyperparameters the API accepts (all optional, task/model-scoped). */
const hyperSchema = {
  alpha: z.number().min(0).max(100).optional(),
  l1Ratio: z.number().min(0).max(1).optional(),
  l2: z.number().min(0).max(100).optional(),
  k: z.number().int().min(1).max(25).optional(),
  maxDepth: z.number().int().min(1).max(12).optional(),
  nEstimators: z.number().int().min(3).max(200).optional(),
  rounds: z.number().int().min(5).max(200).optional(),
  learningRate: z.number().min(0.005).max(1).optional(),
  hidden: z.number().int().min(2).max(64).optional(),
  epochs: z.number().int().min(10).max(2000).optional(),
  lr: z.number().min(0.001).max(1).optional(),
  C: z.number().min(0.01).max(100).optional(),
};

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    model: z.string().min(1).max(64),
    testSize: z.number().min(0.1).max(0.5),
    cv: cvSchema,
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
    ...hyperSchema,
    /* Uploaded-CSV mode: when `csv` is present it replaces the seeded
     * generator entirely (samples/noise/seed are ignored). */
    csv: z.string().min(4).max(262_144).optional(),
    csvName: z.string().min(1).max(128).optional(),
    /* Regression only: multi-feature picks (≥2 → multivariate model). */
    featureColumns: z.array(z.string().min(1).max(128)).max(8).optional(),
    featureColumn: z.string().min(1).max(128).optional(),
    featureColumn2: z.string().min(1).max(128).optional(),
    targetColumn: z.string().min(1).max(128).optional(),
  })
  .superRefine((val, ctx) => {
    if (val.task === "regression" && !(REGRESSION_MODELS as readonly string[]).includes(val.model)) {
      ctx.addIssue({
        code: "custom",
        message: `Regression model must be one of: ${REGRESSION_MODELS.join(", ")}.`,
      });
    }
    if (
      val.task === "classification" &&
      !(CLASSIFICATION_MODELS as readonly string[]).includes(val.model)
    ) {
      ctx.addIssue({
        code: "custom",
        message: `Classification model must be one of: ${CLASSIFICATION_MODELS.join(", ")}.`,
      });
    }
    /* kNN: odd k only, so majority votes can never tie. */
    if (val.k !== undefined && val.k % 2 === 0) {
      ctx.addIssue({
        code: "custom",
        message: "k must be an odd integer between 1 and 25 (even k can tie).",
      });
    }
  });

type TrainBody = {
  task: "regression" | "classification";
  samples: number;
  noise: number;
  model: string;
  testSize: number;
  cv: 0 | 3 | 5 | 10;
  seed: number;
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
};

/** Hyperparameters with per-model defaults applied. */
interface Hyper {
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
}

function resolveHyper(req: TrainBody): Hyper {
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
  };
}

/** Deterministic RNG stream for model internals (bagging / MLP init).
 *  Slot 0 = the main run, slots 1..k = the CV folds — every training run
 *  gets its own stream so identical requests stay byte-identical. */
function modelRng(effSeed: number, slot: number): () => number {
  return mulberry32((effSeed + 104729 + 6151 * slot) >>> 0);
}

function num(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Count numeric columns without importing the resolver (for mode choice). */
function numericCandidates(table: { columns: string[]; rows: string[][] }): number {
  let count = 0;
  for (let c = 0; c < table.columns.length; c++) {
    let nonEmpty = 0;
    let parseable = 0;
    for (const row of table.rows) {
      const cell = row[c] ?? "";
      if (cell === "") continue;
      nonEmpty++;
      if (Number.isFinite(Number(cell))) parseable++;
    }
    if (nonEmpty > 0 && parseable / nonEmpty >= 0.9) count++;
  }
  return count;
}

/** Min/max via loop — safe for very large CSVs (spread would overflow). */
function extent(values: number[]): [number, number] {
  let lo = values[0] ?? 0;
  let hi = lo;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

/** k-fold CV helper: runs `fitFold` per fold with its own RNG slot and
 *  reduces fold values to mean ± population std. */
function runCv(
  n: number,
  k: number,
  effSeed: number,
  metricName: string,
  fitFold: (fold: number[], trainIdx: number[], slot: number) => number
): MLTrainResponse["cv"] {
  if (k <= 1) return undefined;
  const cvRand = mulberry32((effSeed + 7919) >>> 0);
  const folds = kFoldSplit(n, k, cvRand);
  const foldResults: CVFold[] = folds.map((fold, i) => {
    const foldSet = new Set(fold);
    const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
    return { fold: i + 1, value: fitFold(fold, trainIdx, i + 1) };
  });
  const values = foldResults.map((f) => f.value);
  return {
    k,
    metricName,
    folds: foldResults,
    mean: mean(values),
    std: populationStd(values),
  };
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const parsed = bodySchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "Invalid request body.",
        detail: parsed.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`),
      },
      { status: 400 }
    );
  }
  const req = parsed.data;

  /* Uploaded-CSV mode: parse + resolve columns up front so bad files/picks
   * surface as actionable 400s instead of opaque 500s. */
  let csvData: CsvRegressionData | CsvClassificationData | CsvMultiRegressionData | undefined;
  let csvName: string | undefined;
  let splitSeed = req.seed;
  if (req.csv !== undefined) {
    csvName = req.csvName ?? "pasted.csv";
    const table = parseCsv(req.csv);
    if (table.rows.length === 0) {
      return NextResponse.json(
        { error: "The uploaded CSV has no data rows below the header." },
        { status: 400 }
      );
    }
    try {
      if (req.task === "regression") {
        /* Multivariate when multiple feature columns are requested (or auto
         * with ≥2 candidate features); single-feature keeps the x↔y chart. */
        const multi =
          (req.featureColumns?.length ?? 0) >= 2 ||
          (req.featureColumns === undefined &&
            !req.featureColumn &&
            numericCandidates(table) >= 3);
        csvData = multi
          ? resolveMultiRegression(table, req.featureColumns, req.targetColumn)
          : resolveRegression(table, req.featureColumn ?? req.featureColumns?.[0], req.targetColumn);
      } else {
        csvData = resolveClassification(
          table,
          req.featureColumn,
          req.featureColumn2,
          req.targetColumn
        );
      }
    } catch (err) {
      return NextResponse.json(
        {
          error: "Could not use the uploaded CSV.",
          detail: [err instanceof Error ? err.message : String(err)],
        },
        { status: 400 }
      );
    }
    /* Same file → same split: derive the split seed from the file content. */
    splitSeed = hashCsvSeed(req.csv);
  }

  try {
    let result: MLTrainResponse;
    if (req.task === "regression") {
      result =
        csvData && "X" in csvData
          ? runMultiRegression(req, csvData, csvName, splitSeed)
          : runRegression(req, csvData as CsvRegressionData | undefined, csvName, splitSeed);
    } else {
      result = runClassification(
        req,
        csvData as CsvClassificationData | undefined,
        csvName,
        splitSeed
      );
    }
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Training failed.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

/* ------------------------------------------------------------------ */
/* Regression                                                          */
/* ------------------------------------------------------------------ */

/** What every regression fit produces: a row predictor plus the honest
 *  parameter summary for the model family (coefficients for the linear
 *  family, normalized impurity-reduction importances for trees/ensembles,
 *  echoed hyperparameters for kNN/MLP). */
interface RegFit {
  predict(xsRaw: number[]): number;
  coefficients?: { intercept: number; coefs: number[]; standardized: boolean };
  featureImportance?: { features: string[]; importance: number[]; measure: "variance" | "gini" };
  modelParams?: Record<string, number>;
}

/** Fit any regression model over a raw (unstandardized) feature matrix. */
function fitRegressionModel(
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
    default:
      throw new Error(`Unknown regression model "${model}".`);
  }
}

function runRegression(
  req: TrainBody,
  csv?: CsvRegressionData,
  csvName?: string,
  splitSeed?: number
): MLTrainResponse {
  let xs: number[];
  let ys: number[];
  let source: CsvSourceMeta | undefined;
  let features: string[];
  let featureNames: string[];
  if (csv) {
    xs = zscore(csv.x); // standardized so coefficients stay comparable
    ys = csv.y; // target kept in original units
    source = {
      kind: "csv",
      fileName: csvName,
      columns: csv.columns,
      featureColumns: [csv.feature],
      targetColumn: csv.target,
      nRows: csv.x.length,
      droppedRows: csv.droppedRows,
    };
    features = [`${csv.feature} (standardized)`];
    featureNames = [csv.feature];
  } else {
    const rand = mulberry32(req.seed);
    ({ xs, ys } = makeRegressionData(req.samples, req.noise, rand));
    features = ["x (standardized)"];
    featureNames = ["x"];
  }
  const effSeed = splitSeed ?? req.seed;
  const rand = mulberry32(effSeed);
  const n = xs.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const hp = resolveHyper(req);
  const isLegacy = req.model === "linear" || req.model === "polynomial" || req.model === "ridge";
  /* Legacy single-feature models keep the exact original call (degree for
   * polynomial, ridge alpha on the diagonal) so responses stay
   * byte-compatible; every other model fits over the 1-column matrix. */
  const degree = req.model === "polynomial" ? 2 : 1;
  const effAlpha = req.model === "ridge" ? hp.alpha : 0;

  const trainX = train.map((i) => [xs[i] ?? 0]);
  const trainY = train.map((i) => ys[i] ?? 0);

  const fit: RegFit = isLegacy
    ? (() => {
        const m = fitLinearRegression(trainX.map((r) => r[0] ?? 0), trainY, {
          alpha: effAlpha,
          degree,
        });
        return {
          predict: (row) => m.predict(row[0] ?? 0),
          coefficients: {
            intercept: m.w[0] ?? 0,
            coefs: m.w.slice(1),
            standardized: true,
          },
        };
      })()
    : fitRegressionModel(req.model, trainX, trainY, hp, modelRng(effSeed, 0), featureNames);

  const yTrue = test.map((i) => ys[i] ?? 0);
  const yPred = test.map((i) => fit.predict([xs[i] ?? 0]));
  const metrics = regressionMetrics(yTrue, yPred);

  // Smooth fitted curve across the full observed x range.
  const [xMin, xMax] = extent(xs);
  const curve = Array.from({ length: 100 }, (_, i) => {
    const x = xMin + ((xMax - xMin) * i) / 99;
    return { x, yhat: fit.predict([x]) };
  });

  // k-fold CV (R² per fold) with an independent, seeded shuffle.
  const cv = runCv(n, req.cv, effSeed, "R²", (fold, trainIdx, slot) => {
    if (isLegacy) {
      const foldModel = fitLinearRegression(
        trainIdx.map((i) => xs[i] ?? 0),
        trainIdx.map((i) => ys[i] ?? 0),
        { alpha: effAlpha, degree }
      );
      const fTrue = fold.map((i) => ys[i] ?? 0);
      const fPred = fold.map((i) => foldModel.predict(xs[i] ?? 0));
      return regressionMetrics(fTrue, fPred).r2;
    }
    const foldFit = fitRegressionModel(
      req.model,
      trainIdx.map((i) => [xs[i] ?? 0]),
      trainIdx.map((i) => ys[i] ?? 0),
      hp,
      modelRng(effSeed, slot),
      featureNames
    );
    const fTrue = fold.map((i) => ys[i] ?? 0);
    const fPred = fold.map((i) => foldFit.predict([xs[i] ?? 0]));
    return regressionMetrics(fTrue, fPred).r2;
  });

  return {
    task: "regression",
    model: req.model,
    metrics: { r2: metrics.r2, rmse: metrics.rmse, mae: metrics.mae },
    cv,
    ...(fit.coefficients ? { coefficients: fit.coefficients } : {}),
    ...(fit.featureImportance ? { featureImportance: fit.featureImportance } : {}),
    ...(fit.modelParams ? { modelParams: fit.modelParams } : {}),
    series: {
      test: test.map((i) => ({ x: xs[i] ?? 0, y: ys[i] ?? 0 })),
      curve,
    },
    datasetMeta: {
      nTotal: n,
      nTrain: train.length,
      nTest: test.length,
      noise: csv ? 0 : req.noise,
      seed: csv ? 0 : req.seed,
      features,
      ...(source ? { source } : {}),
    },
  };
}

/* ------------------------------------------------------------------ */
/* Multivariate regression (uploaded CSV, k ≥ 2 features)              */
/* ------------------------------------------------------------------ */

/**
 * Multivariate CSV regression. The chart becomes predicted-vs-actual:
 * series.test carries {x: ŷ, y: actual} and `curve` is the perfect-prediction
 * identity line, so the existing chart renderer works unchanged.
 */
function runMultiRegression(
  req: TrainBody,
  csv: CsvMultiRegressionData,
  csvName: string | undefined,
  splitSeed: number
): MLTrainResponse {
  const effSeed = splitSeed ?? req.seed;
  const rand = mulberry32(effSeed);
  const n = csv.y.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const hp = resolveHyper(req);
  const trainX = train.map((i) => csv.X[i] ?? []);
  const trainY = train.map((i) => csv.y[i] ?? 0);
  const fit = fitRegressionModel(req.model, trainX, trainY, hp, modelRng(effSeed, 0), csv.features);

  const yTrue = test.map((i) => csv.y[i] ?? 0);
  const yPred = test.map((i) => fit.predict(csv.X[i] ?? []));
  const metrics = regressionMetrics(yTrue, yPred);

  /* Perfect-prediction identity line over the combined ŷ/y extent. */
  const all = [...yTrue, ...yPred];
  const [lo, hi] = extent(all);
  const curve = Array.from({ length: 100 }, (_, i) => {
    const v = lo + ((hi - lo) * i) / 99;
    return { x: v, yhat: v };
  });

  const source: CsvSourceMeta = {
    kind: "csv",
    fileName: csvName,
    columns: csv.columns,
    featureColumns: csv.features,
    targetColumn: csv.target,
    nRows: n,
    droppedRows: csv.droppedRows,
  };

  // k-fold CV (R² per fold) with an independent, seeded shuffle.
  const cv = runCv(n, req.cv, effSeed, "R²", (fold, trainIdx, slot) => {
    const foldFit = fitRegressionModel(
      req.model,
      trainIdx.map((i) => csv.X[i] ?? []),
      trainIdx.map((i) => csv.y[i] ?? 0),
      hp,
      modelRng(effSeed, slot),
      csv.features
    );
    const fTrue = fold.map((i) => csv.y[i] ?? 0);
    const fPred = fold.map((i) => foldFit.predict(csv.X[i] ?? []));
    return regressionMetrics(fTrue, fPred).r2;
  });

  return {
    task: "regression",
    model: req.model,
    metrics: { r2: metrics.r2, rmse: metrics.rmse, mae: metrics.mae },
    cv,
    ...(fit.coefficients ? { coefficients: fit.coefficients } : {}),
    ...(fit.featureImportance ? { featureImportance: fit.featureImportance } : {}),
    ...(fit.modelParams ? { modelParams: fit.modelParams } : {}),
    series: {
      test: yPred.map((yhat, i) => ({ x: yhat, y: yTrue[i] ?? 0 })),
      curve,
    },
    datasetMeta: {
      nTotal: n,
      nTrain: train.length,
      nTest: test.length,
      noise: 0,
      seed: 0,
      features: csv.features.map((f) => `${f} (standardized)`),
      source,
    },
  };
}

/* ------------------------------------------------------------------ */
/* Classification                                                      */
/* ------------------------------------------------------------------ */

/** What every classifier produces: a hard 0/1 prediction over raw feature
 *  pairs plus the honest parameter summary for the model family. */
interface ClsFit {
  predict01(xRaw: number, yRaw: number): 0 | 1;
  coefficients?: { intercept: number; coefs: number[]; standardized: boolean };
  featureImportance?: { features: string[]; importance: number[]; measure: "variance" | "gini" };
  modelParams?: Record<string, number>;
}

/** Fit any classification model over raw (unstandardized) 2-feature rows. */
function fitClassificationModel(
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

function runClassification(
  req: TrainBody,
  csv?: CsvClassificationData,
  csvName?: string,
  splitSeed?: number
): MLTrainResponse {
  let pts: Array<{ x: number; y: number; label: number }>;
  let source: CsvSourceMeta | undefined;
  let features: string[];
  let featureNames: string[];
  if (csv) {
    // Standardize both features (same space the logistic trainer expects).
    const { x1, x2, label } = csv;
    const sx1 = zscore(x1);
    const sx2 = zscore(x2);
    pts = sx1.map((x, i) => ({ x, y: sx2[i] ?? 0, label: label[i] ?? 0 }));
    source = {
      kind: "csv",
      fileName: csvName,
      columns: csv.columns,
      featureColumns: [csv.feature1, csv.feature2],
      targetColumn: csv.target,
      nRows: csv.x1.length,
      droppedRows: csv.droppedRows,
      labelMap: csv.labelMap,
    };
    features = [`${csv.feature1} (standardized)`, `${csv.feature2} (standardized)`];
    featureNames = [csv.feature1, csv.feature2];
  } else {
    const genRand = mulberry32(req.seed);
    pts = makeMoons(req.samples, req.noise, genRand);
    features = ["x1 (standardized)", "x2 (standardized)"];
    featureNames = ["x1", "x2"];
  }
  const effSeed = splitSeed ?? req.seed;
  const rand = mulberry32(effSeed);
  const n = pts.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const hp = resolveHyper(req);
  const trainX = train.map((i) => [pts[i]?.x ?? 0, pts[i]?.y ?? 0]);
  const trainLabels = train.map((i) => pts[i]?.label ?? 0);
  const fit = fitClassificationModel(
    req.model,
    trainX,
    trainLabels,
    hp,
    modelRng(effSeed, 0),
    featureNames
  );

  const yTrue = test.map((i) => pts[i]?.label ?? 0);
  const yPred = test.map((i) => fit.predict01(pts[i]?.x ?? 0, pts[i]?.y ?? 0));
  const metrics = classificationMetrics(yTrue, yPred);

  // Decision boundary: dense grid classified by the trained model.
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const [xMin, xMax] = extent(xs);
  const [yMin, yMax] = extent(ys);
  const G = 60;
  const boundary: Array<{ x: number; y: number; pred: number }> = [];
  for (let gi = 0; gi < G; gi++) {
    for (let gj = 0; gj < G; gj++) {
      const x = (xMin - 0.5) + ((xMax + 0.5 - (xMin - 0.5)) * gi) / (G - 1);
      const y = (yMin - 0.5) + ((yMax + 0.5 - (yMin - 0.5)) * gj) / (G - 1);
      boundary.push({ x, y, pred: fit.predict01(x, y) });
    }
  }

  // k-fold CV (accuracy per fold).
  const cv = runCv(n, req.cv, effSeed, "Accuracy", (fold, trainIdx, slot) => {
    const foldFit = fitClassificationModel(
      req.model,
      trainIdx.map((i) => [pts[i]?.x ?? 0, pts[i]?.y ?? 0]),
      trainIdx.map((i) => pts[i]?.label ?? 0),
      hp,
      modelRng(effSeed, slot),
      featureNames
    );
    let correct = 0;
    for (const i of fold) {
      const p = pts[i] ?? { x: 0, y: 0, label: 0 };
      const pred = foldFit.predict01(p.x, p.y);
      if (pred === p.label) correct++;
    }
    return correct / fold.length;
  });

  return {
    task: "classification",
    model: req.model,
    metrics: {
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1: metrics.f1,
    },
    cv,
    ...(fit.coefficients ? { coefficients: fit.coefficients } : {}),
    ...(fit.featureImportance ? { featureImportance: fit.featureImportance } : {}),
    ...(fit.modelParams ? { modelParams: fit.modelParams } : {}),
    series: {
      test: test.map((i) => ({
        x: pts[i]?.x ?? 0,
        y: pts[i]?.y ?? 0,
        label: pts[i]?.label ?? 0,
      })),
      boundary,
    },
    datasetMeta: {
      nTotal: n,
      nTrain: train.length,
      nTest: test.length,
      noise: csv ? 0 : req.noise,
      seed: csv ? 0 : req.seed,
      features,
      ...(source ? { source } : {}),
    },
  };
}

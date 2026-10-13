import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  hashCsvSeed,
  parseCsv,
  resolveClassification,
  resolveRegression,
  zscore,
  type CsvClassificationData,
  type CsvRegressionData,
} from "@/lib/csv";
import {
  classificationMetrics,
  fitLinearRegression,
  fitLogisticRegression,
  kFoldSplit,
  makeMoons,
  makeRegressionData,
  mean,
  mulberry32,
  populationStd,
  regressionMetrics,
  shuffledSplit,
} from "@/lib/ml-math";
import type { CVFold, CsvSourceMeta, MLTrainResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const cvSchema = z.union([z.literal(0), z.literal(3), z.literal(5), z.literal(10)]);

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    model: z.string().min(1).max(64),
    alpha: z.number().min(0).max(100).optional(),
    l2: z.number().min(0).max(100).optional(),
    testSize: z.number().min(0.1).max(0.5),
    cv: cvSchema,
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
    /* Uploaded-CSV mode: when `csv` is present it replaces the seeded
     * generator entirely (samples/noise/seed are ignored). */
    csv: z.string().min(4).max(262_144).optional(),
    csvName: z.string().min(1).max(128).optional(),
    featureColumn: z.string().min(1).max(128).optional(),
    featureColumn2: z.string().min(1).max(128).optional(),
    targetColumn: z.string().min(1).max(128).optional(),
  })
  .superRefine((val, ctx) => {
    if (val.task === "regression" && !["linear", "polynomial", "ridge"].includes(val.model)) {
      ctx.addIssue({
        code: "custom",
        message: "Regression model must be one of: linear, polynomial, ridge.",
      });
    }
    if (val.task === "classification" && val.model !== "logistic") {
      ctx.addIssue({
        code: "custom",
        message: "Classification model must be: logistic.",
      });
    }
  });

function num(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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
  let csvData: CsvRegressionData | CsvClassificationData | undefined;
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
      csvData =
        req.task === "regression"
          ? resolveRegression(table, req.featureColumn, req.targetColumn)
          : resolveClassification(
              table,
              req.featureColumn,
              req.featureColumn2,
              req.targetColumn
            );
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
      result = runRegression(
        { ...req, task: "regression" },
        num(req.alpha, 1),
        csvData as CsvRegressionData | undefined,
        csvName,
        splitSeed
      );
    } else {
      result = runClassification(
        { ...req, task: "classification" },
        num(req.l2, 1),
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

type RegressionBody = {
  task: "regression";
  samples: number;
  noise: number;
  model: string;
  testSize: number;
  cv: 0 | 3 | 5 | 10;
  seed: number;
};

function runRegression(
  req: RegressionBody,
  alpha: number,
  csv?: CsvRegressionData,
  csvName?: string,
  splitSeed?: number
): MLTrainResponse {
  let xs: number[];
  let ys: number[];
  let source: CsvSourceMeta | undefined;
  let features: string[];
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
  } else {
    const rand = mulberry32(req.seed);
    ({ xs, ys } = makeRegressionData(req.samples, req.noise, rand));
    features = ["x (standardized)"];
  }
  const effSeed = splitSeed ?? req.seed;
  const rand = mulberry32(effSeed);
  const n = xs.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const degree = req.model === "polynomial" ? 2 : 1;
  const effAlpha = req.model === "ridge" ? alpha : 0;
  const model = fitLinearRegression(
    train.map((i) => xs[i] ?? 0),
    train.map((i) => ys[i] ?? 0),
    { alpha: effAlpha, degree }
  );

  const yTrue = test.map((i) => ys[i] ?? 0);
  const yPred = test.map((i) => model.predict(xs[i] ?? 0));
  const metrics = regressionMetrics(yTrue, yPred);

  // Smooth fitted curve across the full observed x range.
  const [xMin, xMax] = extent(xs);
  const curve = Array.from({ length: 100 }, (_, i) => {
    const x = xMin + ((xMax - xMin) * i) / 99;
    return { x, yhat: model.predict(x) };
  });

  // k-fold CV (R² per fold) with an independent, seeded shuffle.
  let cv: MLTrainResponse["cv"];
  if (req.cv > 1) {
    const cvRand = mulberry32((effSeed + 7919) >>> 0);
    const folds = kFoldSplit(n, req.cv, cvRand);
    const foldResults: CVFold[] = folds.map((fold, i) => {
      const foldSet = new Set(fold);
      const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
      const foldModel = fitLinearRegression(
        trainIdx.map((i) => xs[i] ?? 0),
        trainIdx.map((i) => ys[i] ?? 0),
        { alpha: effAlpha, degree }
      );
      const fTrue = fold.map((i) => ys[i] ?? 0);
      const fPred = fold.map((i) => foldModel.predict(xs[i] ?? 0));
      return { fold: i + 1, value: regressionMetrics(fTrue, fPred).r2 };
    });
    const values = foldResults.map((f) => f.value);
    cv = { k: req.cv, metricName: "R²", folds: foldResults, mean: mean(values), std: populationStd(values) };
  }

  return {
    task: "regression",
    model: req.model,
    metrics: { r2: metrics.r2, rmse: metrics.rmse, mae: metrics.mae },
    cv,
    coefficients: {
      intercept: model.w[0] ?? 0,
      coefs: model.w.slice(1),
      standardized: true,
    },
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
/* Classification                                                      */
/* ------------------------------------------------------------------ */

type ClassificationBody = {
  task: "classification";
  samples: number;
  noise: number;
  model: string;
  testSize: number;
  cv: 0 | 3 | 5 | 10;
  seed: number;
};

function runClassification(
  req: ClassificationBody,
  l2: number,
  csv?: CsvClassificationData,
  csvName?: string,
  splitSeed?: number
): MLTrainResponse {
  let pts: Array<{ x: number; y: number; label: number }>;
  let source: CsvSourceMeta | undefined;
  let features: string[];
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
  } else {
    const genRand = mulberry32(req.seed);
    pts = makeMoons(req.samples, req.noise, genRand);
    features = ["x1 (standardized)", "x2 (standardized)"];
  }
  const effSeed = splitSeed ?? req.seed;
  const rand = mulberry32(effSeed);
  const n = pts.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const model = fitLogisticRegression(
    train.map((i) => pts[i] ?? { x: 0, y: 0, label: 0 }),
    { l2, lr: 0.1, epochs: 800 }
  );

  const yTrue = test.map((i) => pts[i]?.label ?? 0);
  const yPred = test.map((i) => (model.predict(pts[i]?.x ?? 0, pts[i]?.y ?? 0) >= 0.5 ? 1 : 0));
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
      boundary.push({ x, y, pred: model.predict(x, y) >= 0.5 ? 1 : 0 });
    }
  }

  // k-fold CV (accuracy per fold).
  let cv: MLTrainResponse["cv"];
  if (req.cv > 1) {
    const cvRand = mulberry32((effSeed + 7919) >>> 0);
    const folds = kFoldSplit(n, req.cv, cvRand);
    const foldResults: CVFold[] = folds.map((fold, i) => {
      const foldSet = new Set(fold);
      const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
      const foldModel = fitLogisticRegression(
        trainIdx.map((i) => pts[i] ?? { x: 0, y: 0, label: 0 }),
        { l2, lr: 0.1, epochs: 800 }
      );
      let correct = 0;
      for (const i of fold) {
        const p = pts[i] ?? { x: 0, y: 0, label: 0 };
        const pred = foldModel.predict(p.x, p.y) >= 0.5 ? 1 : 0;
        if (pred === p.label) correct++;
      }
      return { fold: i + 1, value: correct / fold.length };
    });
    const values = foldResults.map((f) => f.value);
    cv = {
      k: req.cv,
      metricName: "Accuracy",
      folds: foldResults,
      mean: mean(values),
      std: populationStd(values),
    };
  }

  return {
    task: "classification",
    model: "logistic",
    metrics: {
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1: metrics.f1,
    },
    cv,
    coefficients: {
      intercept: model.bias,
      coefs: [...model.w],
      standardized: true,
    },
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

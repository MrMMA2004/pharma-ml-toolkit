import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
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
import type { CVFold, MLTrainResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const cvSchema = z.union([z.literal(0), z.literal(3), z.literal(5), z.literal(10)]);

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000),
    noise: z.number().min(0).max(2),
    model: z.string().min(1).max(64),
    alpha: z.number().min(0).max(100).optional(),
    l2: z.number().min(0).max(100).optional(),
    testSize: z.number().min(0.1).max(0.5),
    cv: cvSchema,
    seed: z.number().int().min(0).max(2_147_483_647),
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

  try {
    let result: MLTrainResponse;
    if (req.task === "regression") {
      result = runRegression({ ...req, task: "regression" }, num(req.alpha, 1));
    } else {
      result = runClassification({ ...req, task: "classification" }, num(req.l2, 1));
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

function runRegression(req: RegressionBody, alpha: number): MLTrainResponse {
  const rand = mulberry32(req.seed);
  const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
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
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const curve = Array.from({ length: 100 }, (_, i) => {
    const x = xMin + ((xMax - xMin) * i) / 99;
    return { x, yhat: model.predict(x) };
  });

  // k-fold CV (R² per fold) with an independent, seeded shuffle.
  let cv: MLTrainResponse["cv"];
  if (req.cv > 1) {
    const cvRand = mulberry32((req.seed + 7919) >>> 0);
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
      noise: req.noise,
      seed: req.seed,
      features: ["x (standardized)"],
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

function runClassification(req: ClassificationBody, l2: number): MLTrainResponse {
  const rand = mulberry32(req.seed);
  const pts = makeMoons(req.samples, req.noise, rand);
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
  const xMin = Math.min(...xs) - 0.5;
  const xMax = Math.max(...xs) + 0.5;
  const yMin = Math.min(...ys) - 0.5;
  const yMax = Math.max(...ys) + 0.5;
  const G = 60;
  const boundary: Array<{ x: number; y: number; pred: number }> = [];
  for (let gi = 0; gi < G; gi++) {
    for (let gj = 0; gj < G; gj++) {
      const x = xMin + ((xMax - xMin) * gi) / (G - 1);
      const y = yMin + ((yMax - yMin) * gj) / (G - 1);
      boundary.push({ x, y, pred: model.predict(x, y) >= 0.5 ? 1 : 0 });
    }
  }

  // k-fold CV (accuracy per fold).
  let cv: MLTrainResponse["cv"];
  if (req.cv > 1) {
    const cvRand = mulberry32((req.seed + 7919) >>> 0);
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
      noise: req.noise,
      seed: req.seed,
      features: ["x1 (standardized)", "x2 (standardized)"],
    },
  };
}

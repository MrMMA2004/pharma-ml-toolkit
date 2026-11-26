import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  hashCsvSeed,
  numericColumnNames,
  parseCsv,
  resolveClassification,
  resolveMultiRegression,
  resolveRegression,
  type CsvClassificationData,
  type CsvMultiRegressionData,
  type CsvRegressionData,
} from "@/lib/csv";
import {
  classificationMetrics,
  fitLinearRegression,
  kFoldSplit,
  makeMoons,
  makeRegressionData,
  mulberry32,
  regressionMetrics,
} from "@/lib/ml-math";
import {
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
  fitClassificationModel,
  fitRegressionModel,
  modelRng,
  resolveHyper,
} from "@/lib/ml-models";
import {
  LC_FRACTIONS,
  lcVerdict,
  learningCurvePoints,
  type LcPoint,
} from "@/lib/learning-curve";
import type { MLLearningCurveResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/* ------------------------------------------------------------------ */
/* Request schema — the full parked train body plus this web-console  */
/* curve's own fold count. This control is local to the Next.js demo.  */
/* ------------------------------------------------------------------ */

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
  epsilon: z.number().min(1.01).max(20).optional(),
  gamma: z.number().min(0.005).max(10).optional(),
  nClusters: z.number().int().min(2).max(8).optional(),
  radiusFactor: z.number().min(0.25).max(6).optional(),
  nComponents: z.number().int().min(1).max(8).optional(),
  lengthScale: z.number().min(0.05).max(10).optional(),
  signalStd: z.number().min(0.05).max(20).optional(),
  noiseStd: z.number().min(0.005).max(5).optional(),
};

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    model: z.string().min(1).max(64),
    testSize: z.number().min(0.1).max(0.5),
    cv: z.union([z.literal(0), z.literal(3), z.literal(5), z.literal(10)]),
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
    folds: z.union([z.literal(3), z.literal(5), z.literal(10)]).default(5),
    ...hyperSchema,
    csv: z.string().min(4).max(262_144).optional(),
    csvName: z.string().min(1).max(128).optional(),
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
    if (val.k !== undefined && val.k % 2 === 0) {
      ctx.addIssue({
        code: "custom",
        message: "k must be an odd integer between 1 and 25 (even k can tie).",
      });
    }
  });

type LcBody = z.infer<typeof bodySchema>;

/* ------------------------------------------------------------------ */
/* POST: rebuild the run's dataset, refit at growing train sizes       */
/* ------------------------------------------------------------------ */

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

  /* The same CSV resolution chain as /api/ml/train, so the curve judges
   * the very rows the trained run saw (same file → same content-hash
   * seed → same fold shuffle as the train run's own CV). */
  let csvData: CsvRegressionData | CsvMultiRegressionData | CsvClassificationData | undefined;
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
        const multi =
          (req.featureColumns?.length ?? 0) >= 2 ||
          (req.featureColumns === undefined &&
            !req.featureColumn &&
            numericColumnNames(table).length >= 3);
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
    splitSeed = hashCsvSeed(req.csv);
  }

  try {
    const result =
      req.task === "regression"
        ? learningCurveRegression(
            req,
            csvData as CsvRegressionData | CsvMultiRegressionData | undefined,
            csvName,
            splitSeed
          )
        : learningCurveClassification(
            req,
            csvData as CsvClassificationData | undefined,
            csvName,
            splitSeed
          );
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Learning curve failed.",
        detail: [err instanceof Error ? err.message : String(err)],
      },
      { status: 500 }
    );
  }
}

/* ------------------------------------------------------------------ */
/* Regression                                                          */
/* ------------------------------------------------------------------ */

function learningCurveRegression(
  req: LcBody,
  csv: CsvRegressionData | CsvMultiRegressionData | undefined,
  csvName: string | undefined,
  splitSeed: number
): MLLearningCurveResponse {
  /* Feature matrix + target in the run's own units. Single-feature
   * keeps a 1-column X so every fitter sees the exact geometry the
   * train run fit on (no pre-scaling — each fitter standardizes on the
   * rows it is given, the leakage discipline the audit established). */
  let X: number[][];
  let y: number[];
  let nFeatures: number;
  let featureNames: string[];
  if (csv && "X" in csv) {
    X = csv.X;
    y = csv.y;
    nFeatures = csv.features.length;
    featureNames = csv.features;
  } else if (csv) {
    X = csv.x.map((v) => [v]);
    y = csv.y;
    nFeatures = 1;
    featureNames = [csv.feature];
  } else {
    const rand = mulberry32(req.seed);
    const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
    X = xs.map((v) => [v]);
    y = ys;
    nFeatures = 1;
    featureNames = ["x"];
  }

  const effSeed = splitSeed ?? req.seed;
  const hp = resolveHyper(req);
  const isLegacy = req.model === "linear" || req.model === "polynomial" || req.model === "ridge";
  const degree = req.model === "polynomial" ? 2 : 1;
  const effAlpha = req.model === "ridge" ? hp.alpha : 0;

  /* Web-console GP convention: when kernel slots are unset, this route
   * runs its deterministic LML grid search once on the full request data,
   * then freezes those values for every size refit. This is not sklearn
   * learning_curve's clone-and-fit behavior (and the full-data selection
   * can include rows later used as validation folds), so do not describe
   * these GP points as sklearn-equivalent or as an independent tuning
   * evaluation. */
  let gpKernel: MLLearningCurveResponse["gpKernel"];
  if (
    req.model === "gaussian_process" &&
    req.lengthScale === undefined &&
    req.signalStd === undefined &&
    req.noiseStd === undefined
  ) {
    const fullFit = fitRegressionModel(
      req.model,
      X,
      y,
      hp,
      modelRng(effSeed, 0),
      featureNames,
      { learnGpKernel: true }
    );
    if (fullFit.gp) {
      hp.lengthScale = fullFit.gp.lengthScale;
      hp.signalStd = fullFit.gp.signalStd;
      hp.noiseStd = fullFit.gp.noiseStd;
      gpKernel = {
        lengthScale: fullFit.gp.lengthScale,
        signalStd: fullFit.gp.signalStd,
        noiseStd: fullFit.gp.noiseStd,
      };
    }
  }

  const n = y.length;
  /* Same fold shuffle the train run's own CV used (effSeed + 7919). */
  const folds = kFoldSplit(n, req.folds, mulberry32((effSeed + 7919) >>> 0));

  const points: LcPoint[] = learningCurvePoints({
    folds: folds.map((fold) => {
      const foldSet = new Set(fold);
      const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
      return { fold, trainIdx };
    }),
    fitAndScore: (trainRows, valRows, slot) => {
      /* ONE fit per (fold, size), then both sides scored with it. */
      const trainX = trainRows.map((i) => X[i] ?? []);
      const trainY = trainRows.map((i) => y[i] ?? 0);
      const predictRow: (row: number[]) => number = isLegacy
        ? (() => {
            const m = fitLinearRegression(
              trainX.map((r) => r[0] ?? 0),
              trainY,
              { alpha: effAlpha, degree }
            );
            return (row: number[]) => m.predict(row[0] ?? 0);
          })()
        : (() => {
            const fit = fitRegressionModel(
              req.model,
              trainX,
              trainY,
              hp,
              modelRng(effSeed, slot),
              featureNames,
              {}
            );
            return fit.predict;
          })();
      const subTrue = trainRows.map((i) => y[i] ?? 0);
      const subPred = trainRows.map((i) => predictRow(X[i] ?? []));
      const foldTrue = valRows.map((i) => y[i] ?? 0);
      const foldPred = valRows.map((i) => predictRow(X[i] ?? []));
      return [regressionMetrics(subTrue, subPred).r2, regressionMetrics(foldTrue, foldPred).r2];
    },
  });

  return finish(req, points, {
    kind: csv ? "csv" : "synthetic",
    fileName: csv ? csvName : undefined,
    nTotal: n,
    nFeatures,
    ...(gpKernel ? { gpKernel } : {}),
  });
}

/* ------------------------------------------------------------------ */
/* Classification                                                      */
/* ------------------------------------------------------------------ */

function learningCurveClassification(
  req: LcBody,
  csv: CsvClassificationData | undefined,
  csvName: string | undefined,
  splitSeed: number
): MLLearningCurveResponse {
  let pts: Array<{ x: number; y: number; label: number }>;
  let featureNames: string[];
  if (csv) {
    const { x1, x2, label } = csv;
    pts = x1.map((x, i) => ({ x, y: x2[i] ?? 0, label: label[i] ?? 0 }));
    featureNames = [csv.feature1, csv.feature2];
  } else {
    const genRand = mulberry32(req.seed);
    pts = makeMoons(req.samples, req.noise, genRand);
    featureNames = ["x1", "x2"];
  }

  const effSeed = splitSeed ?? req.seed;
  const hp = resolveHyper(req);

  /* GP classification with (ℓ, σf) unset: same one-time resolution —
   * the Laplace-lml grid search runs once on the full data, then every
   * refit clones the learned kernel. */
  let gpClsKernel: MLLearningCurveResponse["gpClsKernel"];
  if (
    req.model === "gaussian_process" &&
    req.lengthScale === undefined &&
    req.signalStd === undefined
  ) {
    const fullFit = fitClassificationModel(
      req.model,
      pts.map((p) => [p.x, p.y]),
      pts.map((p) => p.label),
      hp,
      modelRng(effSeed, 0),
      featureNames,
      { learnGpClsKernel: true }
    );
    if (fullFit.gpCls) {
      hp.lengthScale = fullFit.gpCls.lengthScale;
      hp.signalStd = fullFit.gpCls.signalStd;
      gpClsKernel = {
        lengthScale: fullFit.gpCls.lengthScale,
        signalStd: fullFit.gpCls.signalStd,
      };
    }
  }

  const n = pts.length;
  const folds = kFoldSplit(n, req.folds, mulberry32((effSeed + 7919) >>> 0));

  const points: LcPoint[] = learningCurvePoints({
    folds: folds.map((fold) => {
      const foldSet = new Set(fold);
      const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
      return { fold, trainIdx };
    }),
    fitAndScore: (trainRows, valRows, slot) => {
      const trainX = trainRows.map((i) => [pts[i]?.x ?? 0, pts[i]?.y ?? 0]);
      const trainLabels = trainRows.map((i) => pts[i]?.label ?? 0);
      const fit = fitClassificationModel(
        req.model,
        trainX,
        trainLabels,
        hp,
        modelRng(effSeed, slot),
        featureNames,
        {}
      );
      const accuracy = (rows: number[]): number => {
        const truth = rows.map((i) => pts[i]?.label ?? 0);
        const predicted = rows.map((i) => fit.predict01(pts[i]?.x ?? 0, pts[i]?.y ?? 0));
        return classificationMetrics(truth, predicted).accuracy;
      };
      return [accuracy(trainRows), accuracy(valRows)];
    },
  });

  return finish(req, points, {
    kind: csv ? "csv" : "synthetic",
    fileName: csv ? csvName : undefined,
    nTotal: n,
    nFeatures: 2,
    ...(gpClsKernel ? { gpClsKernel } : {}),
  });
}

/* ------------------------------------------------------------------ */
/* Shared tail: gap verdict + response assembly                        */
/* ------------------------------------------------------------------ */

function finish(
  req: LcBody,
  points: LcPoint[],
  meta: {
    kind: "synthetic" | "csv";
    fileName?: string;
    nTotal: number;
    nFeatures: number;
    gpKernel?: MLLearningCurveResponse["gpKernel"];
    gpClsKernel?: MLLearningCurveResponse["gpClsKernel"];
  }
): MLLearningCurveResponse {
  const last = points[points.length - 1];
  const gap = last ? last.trainMean - last.valMean : 0;
  return {
    task: req.task,
    model: req.model,
    metricName: req.task === "regression" ? "R²" : "Accuracy",
    folds: req.folds,
    fractions: [...LC_FRACTIONS],
    points,
    gap,
    verdict: lcVerdict(gap),
    datasetMeta: {
      nTotal: meta.nTotal,
      nFeatures: meta.nFeatures,
      kind: meta.kind,
      ...(meta.fileName !== undefined ? { fileName: meta.fileName } : {}),
    },
    ...(meta.gpKernel ? { gpKernel: meta.gpKernel } : {}),
    ...(meta.gpClsKernel ? { gpClsKernel: meta.gpClsKernel } : {}),
  };
}

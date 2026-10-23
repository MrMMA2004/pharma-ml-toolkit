import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { hashCsvSeed, parseCsv, resolveClassification, resolveMultiRegression } from "@/lib/csv";
import { mulberry32, shuffledSplit } from "@/lib/ml-math";
import {
  fitClassificationModel,
  fitRegressionModel,
  modelRng,
  resolveHyper,
} from "@/lib/ml-models";
import { applicabilityDomain, type AdRow, type AdSummary } from "@/lib/ml-ad";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/* ------------------------------------------------------------------ */
/* Request schema — the train body's CSV half plus the AD knobs        */
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
  noiseStd: z.number().min(0.005).max(20).optional(),
};

const bodySchema = z.object({
  task: z.enum(["regression", "classification"]),
  csv: z.string().min(1),
  csvName: z.string().optional(),
  model: z.string().min(1),
  testSize: z.number().min(0.05).max(0.5).default(0.2),
  featureColumns: z.array(z.string()).optional(),
  featureColumn: z.string().optional(),
  featureColumn2: z.string().optional(),
  targetColumn: z.string().optional(),
  /** AD knobs (the upload's k=5 / p95 / 30-model convention). */
  adK: z.number().int().min(1).max(20).default(5),
  adPercentile: z.number().min(50).max(99.9).default(95),
  adModels: z.number().int().min(0).max(60).default(20),
  ...hyperSchema,
});

type AdBody = z.infer<typeof bodySchema>;

export interface MLAdResponse {
  task: "regression" | "classification";
  model: string;
  summary: AdSummary;
  rows: AdRow[];
  trainMean: number[];
  nTrain: number;
  featureNames: string[];
  /** Why the bootstrap half was skipped (classification / adModels 0). */
  bootstrapNote: string | null;
}

/* ------------------------------------------------------------------ */
/* Handler                                                             */
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
  const req: AdBody = parsed.data;

  /* Same file -> same split (the train route's content-hash convention):
   * the AD verdict names the very held-out rows the parity chart drew. */
  const effSeed = hashCsvSeed(req.csv);
  const table = parseCsv(req.csv);
  if (table.rows.length === 0) {
    return NextResponse.json(
      { error: "The uploaded CSV has no data rows below the header." },
      { status: 400 }
    );
  }

  try {
    if (req.task === "regression") {
      const csv = resolveMultiRegression(table, req.featureColumns, req.targetColumn);
      return NextResponse.json(runRegressionAd(req, csv, effSeed));
    }
    const csv = resolveClassification(
      table,
      req.featureColumn,
      req.featureColumn2,
      req.targetColumn
    );
    return NextResponse.json(runClassificationAd(req, csv, effSeed));
  } catch (err) {
    return NextResponse.json(
      {
        error: "Could not run the applicability-domain analysis.",
        detail: [err instanceof Error ? err.message : String(err)],
      },
      { status: 500 }
    );
  }
}

/* ------------------------------------------------------------------ */
/* Regression — the upload's full workflow (distance + bootstrap)      */
/* ------------------------------------------------------------------ */

function runRegressionAd(
  req: AdBody,
  csv: { X: number[][]; y: number[]; features: string[] },
  effSeed: number
): MLAdResponse {
  const rand = mulberry32(effSeed);
  const n = csv.y.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const k = Math.min(req.adK, train.length - 1);
  if (k < 1) {
    throw new Error(
      `k=${req.adK} needs at least ${req.adK + 1} training rows; only ${train.length} available.`
    );
  }
  const hp = resolveHyper(req);
  const directives =
    req.lengthScale === undefined && req.signalStd === undefined && req.noiseStd === undefined
      ? { learnGpKernel: true }
      : {};
  const trainX = train.map((i) => csv.X[i] ?? []);
  const trainY = train.map((i) => csv.y[i] ?? 0);
  const testX = test.map((i) => csv.X[i] ?? []);
  const names = test.map((i) => `held-out #${i}`);

  const outcome = applicabilityDomain({
    XTrain: trainX,
    yTrain: trainY,
    XTest: testX,
    names,
    k,
    percentile: req.adPercentile,
    nModels: req.adModels,
    memberRand: (member) => modelRng(effSeed, 100 + member),
    refit: (XBoot, yBoot) => {
      const fit = fitRegressionModel(
        req.model,
        XBoot,
        yBoot,
        hp,
        modelRng(effSeed, 0),
        csv.features,
        directives
      );
      return { predict: (row: number[]) => fit.predict(row) };
    },
  });

  return {
    task: "regression",
    model: req.model,
    summary: outcome.summary,
    rows: outcome.rows,
    trainMean: outcome.trainMean,
    nTrain: train.length,
    featureNames: csv.features,
    bootstrapNote:
      req.adModels === 0 ? "kNN-only quick pass (bootstrap models = 0)" : null,
  };
}

/* ------------------------------------------------------------------ */
/* Classification — the distance verdict; labels carry no honest std   */
/* ------------------------------------------------------------------ */

function runClassificationAd(
  req: AdBody,
  csv: { x1: number[]; x2: number[]; label: Array<0 | 1>; feature1: string; feature2: string },
  effSeed: number
): MLAdResponse {
  const rand = mulberry32(effSeed);
  const pts = csv.x1.map((x, i) => ({ x, y: csv.x2[i] ?? 0, label: csv.label[i] ?? 0 }));
  const n = pts.length;
  const { train, test } = shuffledSplit(n, req.testSize, rand);

  const k = Math.min(req.adK, train.length - 1);
  if (k < 1) {
    throw new Error(
      `k=${req.adK} needs at least ${req.adK + 1} training rows; only ${train.length} available.`
    );
  }
  const trainX = train.map((i) => [pts[i]?.x ?? 0, pts[i]?.y ?? 0]);
  const testX = test.map((i) => [pts[i]?.x ?? 0, pts[i]?.y ?? 0]);
  const names = test.map((i) => `held-out #${i}`);

  /* Fit once so a broken model surfaces HERE, not silently. */
  const hp = resolveHyper(req);
  const directives: { learnGpClsKernel?: boolean } =
    req.lengthScale === undefined && req.signalStd === undefined
      ? { learnGpClsKernel: true }
      : {};
  fitClassificationModel(
    req.model,
    trainX,
    train.map((i) => pts[i]?.label ?? 0),
    hp,
    modelRng(effSeed, 0),
    [csv.feature1, csv.feature2],
    directives
  );

  /* Distance-only: the classifier's hard 0/1 votes have no honest
   * per-row standard deviation (the Python side refuses them too);
   * the domain verdict still applies. */
  const outcome = applicabilityDomain({
    XTrain: trainX,
    yTrain: train.map((i) => pts[i]?.label ?? 0),
    XTest: testX,
    names,
    k,
    percentile: req.adPercentile,
    nModels: 0,
    memberRand: () => mulberry32(0),
    refit: () => {
      throw new Error("unreachable: nModels 0 never refits");
    },
  });

  return {
    task: "classification",
    model: req.model,
    summary: outcome.summary,
    rows: outcome.rows,
    trainMean: outcome.trainMean,
    nTrain: train.length,
    featureNames: [csv.feature1, csv.feature2],
    bootstrapNote:
      "classification models predict hard labels — the bootstrap uncertainty half " +
      "needs numeric predictions, so this is the distance-only verdict " +
      "(train a regression model for the full workflow)",
  };
}

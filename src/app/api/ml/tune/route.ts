import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  hashCsvSeed,
  numericColumnNames,
  parseCsv,
  resolveClassification,
  resolveMultiRegression,
  resolveRegression,
  zscore,
} from "@/lib/csv";
import {
  classificationMetrics,
  kFoldSplit,
  makeMoons,
  makeRegressionData,
  mean,
  mulberry32,
  regressionMetrics,
  shuffledSplit,
} from "@/lib/ml-math";
import {
  CLS_TUNE_SPACES,
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
  REG_TUNE_SPACES,
  decodeTuneIndex,
  fitClassificationModel,
  fitRegressionModel,
  modelRng,
  resolveHyper,
  tuneSpaceSize,
  type HyperSource,
  type TuneChoice,
} from "@/lib/ml-models";
import type { CsvSourceMeta, MLTuneResponse, MLTuneTrial } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    model: z.string().min(1).max(64),
    maxTrials: z.number().int().min(4).max(40).default(12),
    cv: z.union([z.literal(3), z.literal(5), z.literal(10)]),
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
    testSize: z.number().min(0.1).max(0.5).default(0.25),
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
  });

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

  /* The search space — models without tunable hyperparameters are rejected
   * up front with an actionable message (mirrors the library's ml tune). */
  const space: TuneChoice[] =
    req.task === "regression"
      ? (REG_TUNE_SPACES[req.model] ?? [])
      : (CLS_TUNE_SPACES[req.model] ?? []);
  if (space.length === 0) {
    return NextResponse.json(
      {
        error: `${req.model} has no tunable hyperparameters — pick a model with a search space.`,
      },
      { status: 400 }
    );
  }
  const spaceSize = tuneSpaceSize(space);

  /* Dataset resolution (identical rules to /api/ml/train — the same
   * resolvers, the same ≥3-candidate multivariate trigger). */
  let xs: number[] = [];
  let ys: number[] = [];
  let X: number[][] | null = null;
  let pts: Array<{ x: number; y: number; label: number }> = [];
  let source: CsvSourceMeta | undefined;
  let features: string[] = [];
  let featureNames: string[] = [];
  let splitSeed = req.seed;

  if (req.csv !== undefined) {
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
        if (multi) {
          const csv = resolveMultiRegression(table, req.featureColumns, req.targetColumn);
          X = csv.X;
          ys = csv.y;
          features = csv.features.map((f) => `${f} (standardized)`);
          featureNames = csv.features;
          source = {
            kind: "csv",
            fileName: req.csvName,
            columns: csv.columns,
            featureColumns: csv.features,
            targetColumn: csv.target,
            nRows: csv.y.length,
            droppedRows: csv.droppedRows,
          };
        } else {
          const csv = resolveRegression(
            table,
            req.featureColumn ?? req.featureColumns?.[0],
            req.targetColumn
          );
          xs = zscore(csv.x);
          ys = csv.y;
          features = [`${csv.feature} (standardized)`];
          featureNames = [csv.feature];
          source = {
            kind: "csv",
            fileName: req.csvName,
            columns: csv.columns,
            featureColumns: [csv.feature],
            targetColumn: csv.target,
            nRows: csv.x.length,
            droppedRows: csv.droppedRows,
          };
        }
      } else {
        const csv = resolveClassification(
          table,
          req.featureColumn,
          req.featureColumn2,
          req.targetColumn
        );
        const sx1 = zscore(csv.x1);
        const sx2 = zscore(csv.x2);
        pts = sx1.map((x, i) => ({ x, y: sx2[i] ?? 0, label: csv.label[i] ?? 0 }));
        features = [`${csv.feature1} (standardized)`, `${csv.feature2} (standardized)`];
        featureNames = [csv.feature1, csv.feature2];
        source = {
          kind: "csv",
          fileName: req.csvName,
          columns: csv.columns,
          featureColumns: [csv.feature1, csv.feature2],
          targetColumn: csv.target,
          nRows: csv.x1.length,
          droppedRows: csv.droppedRows,
          labelMap: csv.labelMap,
        };
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
  } else {
    if (req.task === "regression") {
      const rand = mulberry32(req.seed);
      ({ xs, ys } = makeRegressionData(req.samples, req.noise, rand));
      features = ["x (standardized)"];
      featureNames = ["x"];
    } else {
      const rand = mulberry32(req.seed);
      pts = makeMoons(req.samples, req.noise, rand);
      features = ["x1 (standardized)", "x2 (standardized)"];
      featureNames = ["x1", "x2"];
    }
  }

  const effSeed = splitSeed ?? req.seed;
  const n = req.task === "regression" ? (X ? ys.length : xs.length) : pts.length;
  if (n < 12) {
    return NextResponse.json(
      { error: "Need at least 12 rows to cross-validate the search trials." },
      { status: 400 }
    );
  }

  /* Row accessors shared by both tasks (single-feature regression wraps
   * xs into a 1-column matrix; multivariate uses the CSV matrix). */
  const row = (i: number): number[] =>
    req.task === "regression"
      ? X
        ? (X[i] ?? [])
        : [xs[i] ?? 0]
      : [pts[i]?.x ?? 0, pts[i]?.y ?? 0];
  const target = (i: number): number =>
    req.task === "regression" ? (ys[i] ?? 0) : (pts[i]?.label ?? 0);

  /* Deterministic CV folds, shared by every trial so scores are directly
   * comparable (same folds, same splits — only the params vary). */
  const cvRand = mulberry32((effSeed + 7919) >>> 0);
  const folds = kFoldSplit(n, req.cv, cvRand);
  const foldSplits = folds.map((fold) => {
    const foldSet = new Set(fold);
    const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
    return { fold, trainIdx };
  });

  const objective = req.task === "regression" ? "R²" : "Accuracy";

  /** Mean CV score of one hyperparameter assignment. */
  function scoreConfig(hpSrc: HyperSource): number {
    const hp = resolveHyper(hpSrc);
    const scores = foldSplits.map(({ fold, trainIdx }, f) => {
      const trainX = trainIdx.map(row);
      const trainY = trainIdx.map(target);
      if (req.task === "regression") {
        const fit = fitRegressionModel(
          req.model,
          trainX,
          trainY,
          hp,
          modelRng(effSeed, f + 1),
          featureNames
        );
        const yTrue = fold.map(target);
        const yPred = fold.map((i) => fit.predict(row(i)));
        return regressionMetrics(yTrue, yPred).r2;
      }
      const fit = fitClassificationModel(
        req.model,
        trainX,
        trainY,
        hp,
        modelRng(effSeed, f + 1),
        featureNames
      );
      const yTrue = fold.map(target);
      const yPred = fold.map((i) => fit.predict01(row(i)[0] ?? 0, row(i)[1] ?? 0));
      return classificationMetrics(yTrue, yPred).accuracy;
    });
    return mean(scores);
  }

  try {
    /* Baseline: the default configuration (resolveHyper defaults — exactly
     * what Train model would use with no overrides). */
    const defaultHp = resolveHyper({});
    const baselineParams = space.reduce<Record<string, number>>((acc, c) => {
      acc[c.key as string] = defaultHp[c.key as keyof typeof defaultHp] as number;
      return acc;
    }, {});
    const baselineScore = scoreConfig(baselineParams);

    /* Trial indices: budget ≥ spaceSize → the whole grid (deterministic
     * order); otherwise seeded sampling without replacement. */
    const exhaustive = req.maxTrials >= spaceSize;
    const trialCount = Math.min(req.maxTrials, spaceSize);
    const indices: number[] = [];
    if (exhaustive) {
      for (let i = 0; i < spaceSize; i++) indices.push(i);
    } else {
      const seen = new Set<number>();
      const sampler = mulberry32((effSeed + 104729) >>> 0);
      while (indices.length < trialCount) {
        const idx = Math.floor(sampler() * spaceSize);
        if (!seen.has(idx)) {
          seen.add(idx);
          indices.push(idx);
        }
      }
    }

    const trials: MLTuneTrial[] = indices.map((idx, t) => {
      const params = decodeTuneIndex(space, idx);
      return { trial: t + 1, params, score: scoreConfig(params) };
    });

    /* Leaderboard: best score first; ties keep the earlier trial ahead. */
    const ranked = [...trials].sort((a, b) => b.score - a.score || a.trial - b.trial);
    const winner = ranked[0] ?? trials[0]!;

    /* Best-config refit on the train split → honest held-out test metric. */
    const splitRand = mulberry32((effSeed + 7919) >>> 0);
    const { train, test } = shuffledSplit(n, req.testSize, splitRand);
    const bestHp = resolveHyper(winner.params);
    let testMetric: number;
    let testMetricName: string;
    if (req.task === "regression") {
      const fit = fitRegressionModel(
        req.model,
        train.map(row),
        train.map(target),
        bestHp,
        modelRng(effSeed, 0),
        featureNames
      );
      const yTrue = test.map(target);
      const yPred = test.map((i) => fit.predict(row(i)));
      testMetric = regressionMetrics(yTrue, yPred).r2;
      testMetricName = "R²";
    } else {
      const fit = fitClassificationModel(
        req.model,
        train.map(row),
        train.map(target),
        bestHp,
        modelRng(effSeed, 0),
        featureNames
      );
      const yTrue = test.map(target);
      const yPred = test.map((i) => fit.predict01(row(i)[0] ?? 0, row(i)[1] ?? 0));
      testMetric = classificationMetrics(yTrue, yPred).accuracy;
      testMetricName = "Accuracy";
    }

    const response: MLTuneResponse = {
      task: req.task,
      model: req.model,
      objective,
      cvK: req.cv,
      maxTrials: req.maxTrials,
      spaceSize,
      exhaustive,
      trials: ranked,
      baseline: { params: baselineParams, score: baselineScore },
      best: {
        params: winner.params,
        cvScore: winner.score,
        testMetric,
        testMetricName,
        improvement: winner.score - baselineScore,
      },
      datasetMeta: {
        nTotal: n,
        features,
        ...(source ? { source } : {}),
      },
    };
    return NextResponse.json(response);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Hyperparameter search failed.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

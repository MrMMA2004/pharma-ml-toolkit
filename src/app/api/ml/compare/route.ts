import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { resolveMlDataset } from "@/lib/ml-data";
import {
  classificationMetrics,
  kFoldSplit,
  mean,
  mulberry32,
  populationStd,
  regressionMetrics,
  shuffledSplit,
} from "@/lib/ml-math";
import {
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
  fitClassificationModel,
  fitRegressionModel,
  modelRng,
  resolveHyper,
} from "@/lib/ml-models";
import type { MLCompareEntry, MLCompareResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * POST /api/ml/compare — train 2–6 models side by side at their default
 * hyperparameters and rank them on the SAME CV folds (the web mirror of
 * `pharmaml.ml.training.compare_models`, the signature workflow of the
 * original rar's per-script comparisons). Every model gets:
 * - the mean ± std CV score over folds shared with every other model,
 * - an honest held-out metric from a refit on the train split,
 * - the wall-clock fit time of that refit.
 */
const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    models: z
      .array(z.string().min(1).max(64))
      .min(2, "Pick at least 2 models to compare.")
      .max(6, "Compare at most 6 models at a time.")
      .refine((models) => new Set(models).size === models.length, {
        message: "Duplicate model names are not allowed.",
      }),
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
    const registry = val.task === "regression" ? REGRESSION_MODELS : CLASSIFICATION_MODELS;
    for (const model of val.models) {
      if (!(registry as readonly string[]).includes(model)) {
        ctx.addIssue({
          code: "custom",
          message: `${
            val.task === "regression" ? "Regression" : "Classification"
          } model must be one of: ${registry.join(", ")}.`,
        });
        break; // one actionable message beats six near-identical ones
      }
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

  /* Dataset resolution — the shared resolver (identical rules to
   * /api/ml/train and /api/ml/tune). */
  const resolved = resolveMlDataset({
    task: req.task,
    samples: req.samples,
    noise: req.noise,
    seed: req.seed,
    csv: req.csv,
    csvName: req.csvName,
    featureColumns: req.featureColumns,
    featureColumn: req.featureColumn,
    featureColumn2: req.featureColumn2,
    targetColumn: req.targetColumn,
  });
  if (!resolved.ok) {
    return NextResponse.json(
      { error: resolved.error, ...(resolved.detail ? { detail: resolved.detail } : {}) },
      { status: 400 }
    );
  }
  const data = resolved.data;

  const effSeed = data.splitSeed;
  const n = data.n;
  if (n < 12) {
    return NextResponse.json(
      { error: "Need at least 12 rows to cross-validate the comparison." },
      { status: 400 }
    );
  }

  const row = (i: number): number[] => data.row(i);
  const target = (i: number): number => data.target(i);

  /* Deterministic CV folds, shared by EVERY model so the ranking is a
   * fair comparison (same folds, same splits — only the model varies). */
  const cvRand = mulberry32((effSeed + 7919) >>> 0);
  const folds = kFoldSplit(n, req.cv, cvRand);
  const foldSplits = folds.map((fold) => {
    const foldSet = new Set(fold);
    const trainIdx = Array.from({ length: n }, (_, j) => j).filter((j) => !foldSet.has(j));
    return { fold, trainIdx };
  });

  const objective = req.task === "regression" ? "R²" : "Accuracy";

  /* Held-out split — the same one every model refits on (the same
   * derivation the tune route uses, so the two leaderboards agree). */
  const splitRand = mulberry32((effSeed + 7919) >>> 0);
  const { train, test } = shuffledSplit(n, req.testSize, splitRand);

  try {
    const entries: MLCompareEntry[] = req.models.map((model) => {
      const hp = resolveHyper({}); // defaults — the tuned-defaults contract

      /* CV score over the shared folds (each fold its own RNG slot,
       * exactly like the tune route's scoreConfig). */
      const foldScores = foldSplits.map(({ fold, trainIdx }, f) => {
        const trainX = trainIdx.map(row);
        const trainY = trainIdx.map(target);
        if (req.task === "regression") {
          const fit = fitRegressionModel(
            model,
            trainX,
            trainY,
            hp,
            modelRng(effSeed, f + 1),
            data.featureNames
          );
          const yTrue = fold.map(target);
          const yPred = fold.map((i) => fit.predict(row(i)));
          return regressionMetrics(yTrue, yPred).r2;
        }
        const fit = fitClassificationModel(
          model,
          trainX,
          trainY,
          hp,
          modelRng(effSeed, f + 1),
          data.featureNames
        );
        const yTrue = fold.map(target);
        const yPred = fold.map((i) => fit.predict01(row(i)[0] ?? 0, row(i)[1] ?? 0));
        return classificationMetrics(yTrue, yPred).accuracy;
      });
      const cvMean = mean(foldScores);
      const cvStd = populationStd(foldScores);

      /* Honest held-out refit (slot 0 — the Train model RNG contract). */
      const t0 = performance.now();
      let testMetric: number;
      if (req.task === "regression") {
        const fit = fitRegressionModel(
          model,
          train.map(row),
          train.map(target),
          hp,
          modelRng(effSeed, 0),
          data.featureNames
        );
        const yTrue = test.map(target);
        const yPred = test.map((i) => fit.predict(row(i)));
        testMetric = regressionMetrics(yTrue, yPred).r2;
      } else {
        const fit = fitClassificationModel(
          model,
          train.map(row),
          train.map(target),
          hp,
          modelRng(effSeed, 0),
          data.featureNames
        );
        const yTrue = test.map(target);
        const yPred = test.map((i) => fit.predict01(row(i)[0] ?? 0, row(i)[1] ?? 0));
        testMetric = classificationMetrics(yTrue, yPred).accuracy;
      }
      const fitMs = performance.now() - t0;

      return {
        model,
        cvMean,
        cvStd,
        testMetric,
        metricName: objective,
        fitMs,
        delta: 0, // filled after ranking
        rank: 0, // filled after ranking
      };
    });

    /* Leaderboard: best CV mean first; ties keep the request order
     * (stable sort), and equal scores share the earlier rank. */
    const ranked = [...entries].sort((a, b) => b.cvMean - a.cvMean);
    ranked.forEach((entry, i) => {
      const tiedWithPrev =
        i > 0 && Math.abs(entry.cvMean - ranked[i - 1]!.cvMean) < 1e-12;
      entry.rank = tiedWithPrev ? ranked[i - 1]!.rank : i + 1;
      entry.delta = entry.cvMean - ranked[0]!.cvMean;
    });

    const response: MLCompareResponse = {
      task: req.task,
      objective,
      cvK: req.cv,
      entries: ranked,
      datasetMeta: {
        nTotal: n,
        features: data.features,
        ...(data.source ? { source: data.source } : {}),
      },
    };
    return NextResponse.json(response);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Model comparison failed.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

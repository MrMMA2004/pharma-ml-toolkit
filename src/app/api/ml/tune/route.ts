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
  TUNE_RESOURCES,
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
    searcher: z.enum(["random", "hyperband"]).default("random"),
    eta: z.union([z.literal(2), z.literal(3), z.literal(4)]).default(3),
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
    /* Refit the winning configuration on the train split → honest
     * held-out test metric (shared by both searchers). */
    const splitRand = mulberry32((effSeed + 7919) >>> 0);
    const { train, test } = shuffledSplit(n, req.testSize, splitRand);
    const refitOnSplit = (params: Record<string, number>): {
      testMetric: number;
      testMetricName: string;
    } => {
      const hp = resolveHyper(params as HyperSource);
      if (req.task === "regression") {
        const fit = fitRegressionModel(
          req.model,
          train.map(row),
          train.map(target),
          hp,
          modelRng(effSeed, 0),
          featureNames
        );
        const yTrue = test.map(target);
        const yPred = test.map((i) => fit.predict(row(i)));
        return { testMetric: regressionMetrics(yTrue, yPred).r2, testMetricName: "R²" };
      }
      const fit = fitClassificationModel(
        req.model,
        train.map(row),
        train.map(target),
        hp,
        modelRng(effSeed, 0),
        featureNames
      );
      const yTrue = test.map(target);
      const yPred = test.map((i) => fit.predict01(row(i)[0] ?? 0, row(i)[1] ?? 0));
      return { testMetric: classificationMetrics(yTrue, yPred).accuracy, testMetricName: "Accuracy" };
    };

    /* The default configuration's params on this space. */
    const defaultHp = resolveHyper({});
    const defaultParams = space.reduce<Record<string, number>>((acc, c) => {
      acc[c.key as string] = defaultHp[c.key as keyof typeof defaultHp] as number;
      return acc;
    }, {});

    if (req.searcher === "hyperband") {
      /* -------------------------------------------------------------- */
      /* Hyperband successive halving (Li et al., 2013) — the web mirror */
      /* of pharmaml.ml.tuner.Hyperband over the grid's resource slot.   */
      /* -------------------------------------------------------------- */
      const resourceSpec = TUNE_RESOURCES[req.model];
      const resourceChoice = resourceSpec
        ? space.find((c) => c.key === (resourceSpec.key as keyof HyperSource))
        : undefined;
      if (!resourceSpec || !resourceChoice) {
        return NextResponse.json(
          {
            error: `Hyperband needs a resource hyperparameter — ${req.model} has none. Pick mlp (epochs), random_forest (estimators) or gradient_boosting (boosting rounds), or switch to RandomSearch.`,
          },
          { status: 400 }
        );
      }

      /* The resource ladder: the slot's grid, ascending. Rungs snap onto
       * it (the largest value ≤ the schedule's target — never above). */
      const grid = [...new Set(resourceChoice.values)].sort((a, b) => a - b);
      const R = grid[grid.length - 1] ?? 1;
      const rMin = grid[0] ?? R;
      const snap = (target: number): number => {
        let v = rMin;
        for (const g of grid) if (g <= target && g > v) v = g;
        return v;
      };
      const eta = req.eta;
      const sMax =
        R > rMin && eta > 1 ? Math.floor(Math.log(R / rMin) / Math.log(eta)) : 0;

      /* Configs come from the non-resource slots (the resource is owned
       * by the schedule). Distinct draws without replacement. */
      const nonResourceSpace = space.filter((c) => c !== resourceChoice);
      const nonResourceSize = tuneSpaceSize(nonResourceSpace);

      const trials: MLTuneTrial[] = [];
      const brackets: Array<{ bracket: number; configs: number; rungs: number[] }> = [];
      const ladderSet = new Set<number>();
      let trialCounter = 0;

      /* Most exploratory bracket first (the classic order). */
      for (let s = sMax; s >= 0; s--) {
        const want = Math.ceil(((sMax + 1) * Math.pow(eta, s)) / (s + 1));
        const nConfigs = Math.max(1, Math.min(want, nonResourceSize || 1));
        const rungs = Array.from({ length: s + 1 }, (_, i) => snap(R / Math.pow(eta, s - i)));
        rungs.forEach((r) => ladderSet.add(r));
        brackets.push({ bracket: s, configs: nConfigs, rungs: [...rungs] });

        /* Seeded draws without replacement from the non-resource grid. */
        const sampler = mulberry32((effSeed + 104729 + 15485863 * (s + 1)) >>> 0);
        const seen = new Set<number>();
        let configs: Array<Record<string, number>> = [];
        let guard = 0;
        while (configs.length < nConfigs && guard < 1000 * nConfigs) {
          guard++;
          const idx = Math.floor(sampler() * nonResourceSize);
          if (nonResourceSize > 1 && seen.has(idx)) continue;
          seen.add(idx);
          configs.push(decodeTuneIndex(nonResourceSpace, idx));
        }

        for (let rung = 0; rung < rungs.length; rung++) {
          const resource = rungs[rung]!;
          const scored: Array<{ score: number; params: Record<string, number>; ref: MLTuneTrial }> =
            [];
          for (const base of configs) {
            const params = { ...base, [resourceSpec.key]: resource };
            const score = scoreConfig(params as HyperSource);
            trialCounter += 1;
            const record: MLTuneTrial = {
              trial: trialCounter,
              params,
              score,
              bracket: s,
              rung,
              resource,
              promoted: false,
            };
            trials.push(record);
            scored.push({ score, params, ref: record });
          }
          if (rung === rungs.length - 1) break;
          /* Promote the top ceil(n / eta) to the next rung. */
          scored.sort((a, b) => b.score - a.score);
          const keep = Math.max(1, Math.ceil(configs.length / eta));
          const survivors = scored.slice(0, keep);
          for (const sv of survivors) sv.ref.promoted = true;
          configs = survivors.map((sv) => {
            const { [resourceSpec.key]: _resource, ...rest } = sv.params;
            return rest;
          });
        }
      }

      const ranked = [...trials].sort((a, b) => b.score - a.score || a.trial - b.trial);
      const winner = ranked[0] ?? trials[0]!;
      const { testMetric, testMetricName } = refitOnSplit(winner.params);

      /* Baseline: the defaults at the FULL resource — the honest
       * apples-to-apples comparison (Train model runs at full budget). */
      const baselineParams = { ...defaultParams, [resourceSpec.key]: R };
      const baselineScore = scoreConfig(baselineParams as HyperSource);

      /* Resource economics: spent = Σ resource over trials; naive = every
       * trial at the full budget (the notebook §5 "epoch economics"). */
      const spent = trials.reduce((acc, t) => acc + (t.resource ?? 0), 0);
      const naive = trials.length * R;
      const savedPct = naive > 0 ? Math.max(0, 1 - spent / naive) : 0;

      const response: MLTuneResponse = {
        task: req.task,
        model: req.model,
        searcher: "hyperband",
        objective,
        cvK: req.cv,
        maxTrials: req.maxTrials,
        spaceSize,
        exhaustive: false,
        trials: ranked,
        baseline: { params: baselineParams, score: baselineScore },
        hyperband: {
          eta,
          resourceKey: resourceSpec.key,
          /* Model-specific label ("boosting rounds" for ada_boost, "trees"
           * only for forests) — the UI prefers this over the generic
           * per-key label. */
          resourceLabel: resourceSpec.label,
          maxResource: R,
          ladder: [...ladderSet].sort((a, b) => a - b),
          brackets,
          budget: { spent, naive, savedPct },
        },
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
    }

    /* Baseline: the default configuration (resolveHyper defaults — exactly
     * what Train model would use with no overrides). */
    const baselineScore = scoreConfig(defaultParams);

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
    const { testMetric, testMetricName } = refitOnSplit(winner.params);

    const response: MLTuneResponse = {
      task: req.task,
      model: req.model,
      searcher: "random",
      objective,
      cvK: req.cv,
      maxTrials: req.maxTrials,
      spaceSize,
      exhaustive,
      trials: ranked,
      baseline: { params: defaultParams, score: baselineScore },
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

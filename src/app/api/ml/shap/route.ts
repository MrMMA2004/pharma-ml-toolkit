import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  hashCsvSeed,
  numericColumnNames,
  parseCsv,
  resolveClassification,
  resolveMultiRegression,
  resolveRegression,
} from "@/lib/csv";
import { makeMoons, makeRegressionData, mulberry32, shuffledSplit } from "@/lib/ml-math";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";
import {
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
} from "@/lib/ml-models";
import type { MLShapResponse, ShapRankRow } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/* ------------------------------------------------------------------ */
/* SHAP feature attribution — separate Next.js web-console endpoint    */
/* ------------------------------------------------------------------ */

/**
 * Mean-|SHAP| attribution for the last training run represented by the
 * web console. The route re-derives the same seeded train/test split from
 * the frozen request and sends held-out rows to the Python tree explainer
 * (`scripts/shap_bridge.py`: build_model -> fit -> shap_summary). The
 * card renders the returned ranking in TypeScript. This interface is
 * separate from the React/Tauri Studio and does not imply feature parity.
 *
 * Supported tree families are decision_tree, random_forest, extra_trees,
 * and gradient_boosting. Non-tree models are refused with an explanatory
 * pointer rather than a silent fallback.
 */

/** The tree families with a pharmaml MODEL_REGISTRY twin (all four ship
 *  as always-available sklearn estimators — no optional dependency). */
const TREE_TWINS = new Set(["decision_tree", "random_forest", "extra_trees", "gradient_boosting"]);

/** Number of ranked features returned for this compact web-console card. */
const DEFAULT_TOP_N = 8;

const hyperSchema = {
  /* Only the tree-relevant hyperparameters are forwarded to the Python
   * twin (mapped nEstimators/rounds -> n_estimators, maxDepth ->
   * max_depth, learningRate -> learning_rate by the bridge). The rest of
   * the train body's hyper schema is accepted-and-ignored so the client
   * can forward the frozen request verbatim. */
  nEstimators: z.number().int().min(3).max(200).optional(),
  rounds: z.number().int().min(5).max(200).optional(),
  maxDepth: z.number().int().min(1).max(12).optional(),
  learningRate: z.number().min(0.005).max(1).optional(),
};

const bodySchema = z
  .object({
    task: z.enum(["regression", "classification"]),
    samples: z.number().int().min(100).max(1000).default(300),
    noise: z.number().min(0).max(2).default(0.4),
    model: z.string().min(1).max(64),
    testSize: z.number().min(0.1).max(0.5).default(0.2),
    /* accepted-and-ignored (the train body carries it; the split
     * re-derivation needs no CV folds) */
    cv: z.union([z.literal(0), z.literal(3), z.literal(5), z.literal(10)]).default(0),
    seed: z.number().int().min(0).max(2_147_483_647).default(42),
    ...hyperSchema,
    csv: z.string().min(4).max(262_144).optional(),
    csvName: z.string().min(1).max(128).optional(),
    featureColumns: z.array(z.string().min(1).max(128)).max(8).optional(),
    featureColumn: z.string().min(1).max(128).optional(),
    featureColumn2: z.string().min(1).max(128).optional(),
    targetColumn: z.string().min(1).max(128).optional(),
    topN: z.number().int().min(1).max(20).optional(),
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
    if (!TREE_TWINS.has(val.model)) {
      ctx.addIssue({
        code: "custom",
        message:
          "SHAP explains tree models — pick decision_tree, random_forest, extra_trees, or " +
          "(regression) gradient_boosting. The model-agnostic read is the permutation-importance " +
          "alternative in the Python package is pharmaml.ml.permutation_feature_importance.",
      });
    }
  });

type ShapBody = z.infer<typeof bodySchema>;

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
  const req: ShapBody = parsed.data;

  /* Re-derive the run's own rows + split — the train route's exact data
   * prep (same generators, same seeds, same content-hash convention),
   * so the explained rows are the run's rows. */
  let xAll: number[][];
  let yAll: number[];
  let featureNames: string[];
  let splitSeed = req.seed;
  let sourceNote: string;

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
        /* The train route's multivariate detection, mirrored exactly. */
        const multi =
          (req.featureColumns?.length ?? 0) >= 2 ||
          (req.featureColumns === undefined &&
            !req.featureColumn &&
            numericColumnNames(table).length >= 3);
        const csv = multi
          ? resolveMultiRegression(table, req.featureColumns, req.targetColumn)
          : resolveRegression(table, req.featureColumn ?? req.featureColumns?.[0], req.targetColumn);
        /* Single-feature keeps the train route's x-array shape; the
         * multivariate resolver returns the X matrix — normalise both to
         * the row-matrix form the bridge takes. */
        if ("X" in csv) {
          xAll = csv.X;
          featureNames = csv.features;
        } else {
          xAll = csv.x.map((v) => [v]);
          featureNames = [csv.feature];
        }
        yAll = csv.y;
      } else {
        const csv = resolveClassification(
          table,
          req.featureColumn,
          req.featureColumn2,
          req.targetColumn
        );
        xAll = csv.x1.map((x, i) => [x, csv.x2[i] ?? 0]);
        yAll = csv.label.map((l) => l);
        featureNames = [csv.feature1, csv.feature2];
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
    /* Same file -> same split (the train route's content-hash convention). */
    splitSeed = hashCsvSeed(req.csv);
    sourceNote = "the uploaded CSV's own split";
  } else if (req.task === "regression") {
    const rand = mulberry32(req.seed);
    const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
    xAll = xs.map((x) => [x]);
    yAll = ys;
    featureNames = ["x"];
    sourceNote = "the seeded synthetic regression data's own split";
  } else {
    const genRand = mulberry32(req.seed);
    const pts = makeMoons(req.samples, req.noise, genRand);
    xAll = pts.map((p) => [p.x, p.y]);
    yAll = pts.map((p) => p.label);
    featureNames = ["x1", "x2"];
    sourceNote = "the seeded synthetic moons' own split";
  }

  /* The run's own split: same seed discipline as the train route. */
  const rand = mulberry32(splitSeed);
  const { train, test } = shuffledSplit(xAll.length, req.testSize, rand);
  if (train.length === 0 || test.length === 0) {
    return NextResponse.json(
      { error: "The split left no rows to explain — lower the test fraction." },
      { status: 400 }
    );
  }
  const xTrain = train.map((i) => xAll[i] ?? []);
  const yTrain = train.map((i) => yAll[i] ?? 0);
  const xTest = test.map((i) => xAll[i] ?? []);

  /* Forward ONLY the tree-relevant hyperparameters the run used. */
  const params: Record<string, number> = {};
  if (req.nEstimators !== undefined) params.nEstimators = req.nEstimators;
  if (req.rounds !== undefined) params.rounds = req.rounds;
  if (req.maxDepth !== undefined) params.maxDepth = req.maxDepth;
  if (req.learningRate !== undefined) params.learningRate = req.learningRate;

  try {
    const bridge = await runPythonBridge<{
      task: string;
      model: string;
      pythonTwin: string;
      top: ShapRankRow[];
      baseValue: number | null;
      nSamples: number;
      nFeatures: number;
      nTrain: number;
      shapVersion: string;
      elapsedSeconds: number;
    }>({
      script: "shap_bridge.py",
      payload: {
        task: req.task,
        model: req.model,
        params,
        featureNames,
        xTrain,
        yTrain,
        xTest,
        topN: req.topN ?? DEFAULT_TOP_N,
      },
    });

    const response: MLShapResponse = {
      task: req.task,
      model: req.model,
      pythonTwin: bridge.pythonTwin,
      top: Array.isArray(bridge.top) ? bridge.top : [],
      baseValue: bridge.baseValue ?? null,
      nSamples: bridge.nSamples,
      nFeatures: bridge.nFeatures,
      nTrain: bridge.nTrain,
      featureNames,
      shapVersion: bridge.shapVersion,
      elapsedSeconds: bridge.elapsedSeconds,
      splitNote: `computed over this run's own ${test.length} held-out rows (${sourceNote}, split seed ${
        req.csv !== undefined ? "hashed from the file content" : req.seed
      }) — the toolkit's Python ${bridge.pythonTwin} twin fitted on the same ${train.length} training rows`,
    };
    return NextResponse.json(response);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      /* The bridge's timeout wording names RDKit; re-voice it for SHAP. */
      if (err instanceof Error && err.message === "BRIDGE_TIMEOUT") {
        return NextResponse.json(
          { error: "The Python SHAP backend timed out — try a smaller dataset." },
          { status: 504 }
        );
      }
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "The SHAP explanation failed." },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  makeMoons,
  makeRegressionData,
  mean,
  mulberry32,
  populationStd,
} from "@/lib/ml-math";
import type {
  DatasetColumnStats,
  DatasetPreviewResponse,
} from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Number of rows shown in the preview table. */
const PREVIEW_ROWS = 10;

const bodySchema = z.object({
  task: z.enum(["regression", "classification"]),
  samples: z.number().int().min(100).max(1000),
  noise: z.number().min(0).max(2),
  seed: z.number().int().min(0).max(2_147_483_647),
});

function columnStats(
  column: string,
  values: number[]
): DatasetColumnStats {
  return {
    column,
    mean: mean(values),
    std: populationStd(values),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

/** Pearson correlation between two equal-length numeric arrays. */
function pearson(a: number[], b: number[]): number {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = (a[i] ?? 0) - ma;
    const y = (b[i] ?? 0) - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  const den = Math.sqrt(da * db);
  return den === 0 ? 0 : num / den;
}

/**
 * POST /api/ml/dataset — preview the deterministic dataset that
 * /api/ml/train would generate for the same task + seed. Uses the exact
 * same generators, so rows and statistics match training data 1:1.
 */
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

  const rand = mulberry32(req.seed);

  if (req.task === "regression") {
    const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
    const rows = Array.from({ length: Math.min(PREVIEW_ROWS, xs.length) }, (_, i) => ({
      i: i + 1,
      x: xs[i] ?? 0,
      y: ys[i] ?? 0,
    }));
    const response: DatasetPreviewResponse = {
      task: "regression",
      columns: ["i", "x", "y"],
      rows,
      nTotal: xs.length,
      stats: [columnStats("x", xs), columnStats("y", ys)],
      correlation: pearson(xs, ys),
      noise: req.noise,
      seed: req.seed,
    };
    return NextResponse.json(response);
  }

  const pts = makeMoons(req.samples, req.noise, rand);
  const rows = Array.from({ length: Math.min(PREVIEW_ROWS, pts.length) }, (_, i) => ({
    i: i + 1,
    x1: pts[i]?.x ?? 0,
    x2: pts[i]?.y ?? 0,
    label: pts[i]?.label ?? 0,
  }));
  const labels = pts.map((p) => p.label);
  const n0 = labels.filter((l) => l === 0).length;
  const response: DatasetPreviewResponse = {
    task: "classification",
    columns: ["i", "x1", "x2", "label"],
    rows,
    nTotal: pts.length,
    stats: [
      columnStats("x1", pts.map((p) => p.x)),
      columnStats("x2", pts.map((p) => p.y)),
    ],
    classBalance: [
      { label: 0, count: n0, pct: (100 * n0) / pts.length },
      { label: 1, count: pts.length - n0, pct: (100 * (pts.length - n0)) / pts.length },
    ],
    noise: req.noise,
    seed: req.seed,
  };
  return NextResponse.json(response);
}

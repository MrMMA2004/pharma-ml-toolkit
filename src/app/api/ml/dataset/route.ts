import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import {
  parseCsv,
  resolveClassification,
  resolveRegression,
  type CsvClassificationData,
  type CsvRegressionData,
} from "@/lib/csv";
import {
  makeMoons,
  makeRegressionData,
  mean,
  mulberry32,
  populationStd,
  regressionTrueY,
} from "@/lib/ml-math";
import type {
  CsvSourceMeta,
  DatasetColumnStats,
  DatasetPreviewResponse,
} from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Number of rows shown in the preview table. */
const PREVIEW_ROWS = 10;

/** Maximum points returned for the overview scatter chart. */
const MAX_POINTS = 300;

/** Evenly strided indices [0, n) capped at `max` samples (deterministic). */
function stridedIndices(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = n / max;
  return Array.from({ length: max }, (_, i) => Math.floor(i * step));
}

/** The noiseless generating line sampled on a 25-point grid over x ∈ [-3, 3]. */
function regressionTrend(): Array<{ x: number; y: number }> {
  return Array.from({ length: 25 }, (_, i) => {
    const x = -3 + (6 * i) / 24;
    return { x, y: regressionTrueY(x) };
  });
}

const bodySchema = z.object({
  task: z.enum(["regression", "classification"]),
  samples: z.number().int().min(100).max(1000).default(300),
  noise: z.number().min(0).max(2).default(0.4),
  seed: z.number().int().min(0).max(2_147_483_647).default(42),
  /* Uploaded-CSV mode: preview the user's own rows/stats instead. */
  csv: z.string().min(4).max(262_144).optional(),
  csvName: z.string().min(1).max(128).optional(),
  featureColumn: z.string().min(1).max(128).optional(),
  featureColumn2: z.string().min(1).max(128).optional(),
  targetColumn: z.string().min(1).max(128).optional(),
});

function columnStats(
  column: string,
  values: number[]
): DatasetColumnStats {
  let min = values[0] ?? 0;
  let max = min;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return {
    column,
    mean: mean(values),
    std: populationStd(values),
    min,
    max,
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

/** Index column name that never collides with a user column. */
function indexColumn(...used: string[]): string {
  return used.includes("i") ? "i#" : "i";
}

/** Build the CSV-source preview (raw column values, strided scatter points). */
function csvPreview(
  task: "regression" | "classification",
  csvName: string | undefined,
  data: CsvRegressionData | CsvClassificationData
): DatasetPreviewResponse {
  if (task === "regression") {
    const d = data as CsvRegressionData;
    const source: CsvSourceMeta = {
      kind: "csv",
      fileName: csvName,
      columns: d.columns,
      featureColumns: [d.feature],
      targetColumn: d.target,
      nRows: d.x.length,
      droppedRows: d.droppedRows,
    };
    const i0 = indexColumn(d.feature, d.target);
    const rows = Array.from({ length: Math.min(PREVIEW_ROWS, d.x.length) }, (_, i) => ({
      [i0]: i + 1,
      [d.feature]: d.x[i] ?? 0,
      [d.target]: d.y[i] ?? 0,
    }));
    const idx = stridedIndices(d.x.length, MAX_POINTS);
    return {
      task,
      columns: [i0, d.feature, d.target],
      rows,
      nTotal: d.x.length,
      stats: [columnStats(d.feature, d.x), columnStats(d.target, d.y)],
      points: idx.map((i) => ({ x: d.x[i] ?? 0, y: d.y[i] ?? 0 })),
      nPoints: idx.length,
      correlation: pearson(d.x, d.y),
      noise: 0,
      seed: 0,
      source,
    };
  }

  const d = data as CsvClassificationData;
  const source: CsvSourceMeta = {
    kind: "csv",
    fileName: csvName,
    columns: d.columns,
    featureColumns: [d.feature1, d.feature2],
    targetColumn: d.target,
    nRows: d.x1.length,
    droppedRows: d.droppedRows,
    labelMap: d.labelMap,
  };
  const i0 = indexColumn(d.feature1, d.feature2, d.target);
  const rows = Array.from({ length: Math.min(PREVIEW_ROWS, d.x1.length) }, (_, i) => ({
    [i0]: i + 1,
    [d.feature1]: d.x1[i] ?? 0,
    [d.feature2]: d.x2[i] ?? 0,
    [d.target]: d.label[i] ?? 0,
  }));
  const idx = stridedIndices(d.x1.length, MAX_POINTS);
  const n1 = d.label.filter((l) => l === 1).length;
  const n0 = d.x1.length - n1;
  return {
    task,
    columns: [i0, d.feature1, d.feature2, d.target],
    rows,
    nTotal: d.x1.length,
    stats: [columnStats(d.feature1, d.x1), columnStats(d.feature2, d.x2)],
    points: idx.map((i) => ({
      x: d.x1[i] ?? 0,
      y: d.x2[i] ?? 0,
      label: d.label[i] ?? 0,
    })),
    nPoints: idx.length,
    classBalance: [
      { label: 0, count: n0, pct: (100 * n0) / d.x1.length },
      { label: 1, count: n1, pct: (100 * n1) / d.x1.length },
    ],
    noise: 0,
    seed: 0,
    source,
  };
}

/**
 * POST /api/ml/dataset — preview the dataset that /api/ml/train would use.
 * Synthetic mode: deterministic generators (same seed → same rows).
 * CSV mode: the user's uploaded rows, stats and scatter (raw column values).
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

  /* Uploaded-CSV mode: resolve + preview the user's own data. */
  if (req.csv !== undefined) {
    const table = parseCsv(req.csv);
    if (table.rows.length === 0) {
      return NextResponse.json(
        { error: "The uploaded CSV has no data rows below the header." },
        { status: 400 }
      );
    }
    try {
      const data =
        req.task === "regression"
          ? resolveRegression(table, req.featureColumn, req.targetColumn)
          : resolveClassification(
              table,
              req.featureColumn,
              req.featureColumn2,
              req.targetColumn
            );
      return NextResponse.json(
        csvPreview(req.task, req.csvName ?? "pasted.csv", data)
      );
    } catch (err) {
      return NextResponse.json(
        {
          error: "Could not preview the uploaded CSV.",
          detail: [err instanceof Error ? err.message : String(err)],
        },
        { status: 400 }
      );
    }
  }

  const rand = mulberry32(req.seed);

  if (req.task === "regression") {
    const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
    const rows = Array.from({ length: Math.min(PREVIEW_ROWS, xs.length) }, (_, i) => ({
      i: i + 1,
      x: xs[i] ?? 0,
      y: ys[i] ?? 0,
    }));
    const points = stridedIndices(xs.length, MAX_POINTS).map((i) => ({
      x: xs[i] ?? 0,
      y: ys[i] ?? 0,
    }));
    const response: DatasetPreviewResponse = {
      task: "regression",
      columns: ["i", "x", "y"],
      rows,
      nTotal: xs.length,
      stats: [columnStats("x", xs), columnStats("y", ys)],
      points,
      nPoints: points.length,
      trend: regressionTrend(),
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
  const points = stridedIndices(pts.length, MAX_POINTS).map((i) => {
    const p = pts[i];
    return { x: p?.x ?? 0, y: p?.y ?? 0, label: p?.label ?? 0 };
  });
  const response: DatasetPreviewResponse = {
    task: "classification",
    columns: ["i", "x1", "x2", "label"],
    rows,
    nTotal: pts.length,
    stats: [
      columnStats("x1", pts.map((p) => p.x)),
      columnStats("x2", pts.map((p) => p.y)),
    ],
    points,
    nPoints: points.length,
    classBalance: [
      { label: 0, count: n0, pct: (100 * n0) / pts.length },
      { label: 1, count: pts.length - n0, pct: (100 * (pts.length - n0)) / pts.length },
    ],
    noise: req.noise,
    seed: req.seed,
  };
  return NextResponse.json(response);
}

import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { makeMoons, makeRegressionData, mulberry32 } from "@/lib/ml-math";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const querySchema = z.object({
  task: z.enum(["regression", "classification"]),
  samples: z.coerce.number().int().min(100).max(1000),
  noise: z.coerce.number().min(0).max(2),
  seed: z.coerce.number().int().min(0).max(2_147_483_647),
});

/** Render a number for CSV: full precision without exponent noise. */
function csvNum(value: number): string {
  if (!Number.isFinite(value)) return "";
  //toFixed(6) keeps rows tidy; strip trailing zeros for smaller files.
  const fixed = value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return fixed === "-0" ? "0" : fixed;
}

/**
 * GET /api/ml/dataset/csv — the full deterministic dataset behind the ML
 * playground (same generators + seed as /api/ml/train) as a CSV download.
 * Query: task, samples (100–1000), noise (0–2), seed.
 */
export async function GET(request: NextRequest) {
  const params = Object.fromEntries(request.nextUrl.searchParams.entries());
  const parsed = querySchema.safeParse(params);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "Invalid query parameters.",
        detail: parsed.error.issues.map(
          (i) => `${i.path.join(".") || "query"}: ${i.message}`
        ),
      },
      { status: 400 }
    );
  }
  const req = parsed.data;
  const rand = mulberry32(req.seed);

  const lines: string[] = [];
  if (req.task === "regression") {
    const { xs, ys } = makeRegressionData(req.samples, req.noise, rand);
    lines.push("i,x,y");
    for (let i = 0; i < xs.length; i++) {
      lines.push(`${i + 1},${csvNum(xs[i] ?? 0)},${csvNum(ys[i] ?? 0)}`);
    }
  } else {
    const pts = makeMoons(req.samples, Math.min(req.noise, 1.5), rand);
    lines.push("i,x1,x2,label");
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      lines.push(
        `${i + 1},${csvNum(p?.x ?? 0)},${csvNum(p?.y ?? 0)},${p?.label ?? 0}`
      );
    }
  }

  const csv = `${lines.join("\n")}\n`;
  const fileName = `pharmaml_${req.task}_n${req.samples}_seed${req.seed}.csv`;

  return new NextResponse(csv, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${fileName}"`,
      "Cache-Control": "no-store",
    },
  });
}

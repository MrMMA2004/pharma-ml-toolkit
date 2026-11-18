import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Round 249 -- the Molecular Dynamics tab's GROMACS export parser: one
 * POST feeds pasted .xvg / .ndx text through the REAL
 * pharmaml.md.study pipeline (the productionized upload/MD_analysis.py
 * script): parsing, kJ->kcal on binding energies, ps->ns on long time
 * axes, 3-point H-bond smoothing, per-metric stats, the cross-metric
 * correlation matrix and the summary CSV -- the same numbers the
 * uploaded script drew as JPEG panels.
 */
const fileSchema = z.object({
  name: z
    .string()
    .min(1, "file name must not be empty")
    .max(80, "file names stay short"),
  content: z.string().min(1, "file content must not be empty").max(200_000),
});

const bodySchema = z.object({
  files: z
    .array(fileSchema)
    .min(1, "pass at least one export file")
    .max(12, "the web parser handles at most 12 files per run"),
  smoothHbonds: z.boolean().optional(),
});

interface BridgeRequest {
  files: { name: string; content: string }[];
  smoothHbonds?: boolean;
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
        detail: parsed.error.issues.map(
          (i) => `${i.path.join(".") || "body"}: ${i.message}`
        ),
      },
      { status: 400 }
    );
  }

  try {
    const result = await runPythonBridge({
      script: "md_bridge.py",
      payload: { op: "xvg", ...(parsed.data as BridgeRequest) },
      timeoutMs: 60_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "MD export parsing failed." },
      { status: 500 }
    );
  }
}

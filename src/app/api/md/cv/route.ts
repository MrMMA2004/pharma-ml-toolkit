import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Round 258 -- the Molecular Dynamics tab's COLVAR playground card:
 * one POST runs the REAL pharmaml.md.cv stack on the deterministic
 * alpha-helix demo -- the pasted plumed.dat snippet is parsed by
 * parse_plumed_cvs (labels, 1-based ATOMS with comma lists and a-b
 * ranges, GROUPA/GROUPB/R_0, the MASS flag; unknown keywords are
 * refused, never silently ignored) and every CV time series, the
 * honest per-CV stats and the exact COLVAR text PLUMED's PRINT would
 * have written come back from the shipped toolkit, so the web demo
 * and the Python API cannot drift apart.
 */
const bodySchema = z.object({
  nFrames: z
    .number()
    .int()
    .min(4, "at least 4 frames")
    .max(200, "the web lab handles at most 200 frames")
    .optional(),
  seed: z.number().int().min(0).max(2 ** 31 - 1).optional(),
  noise: z.number().min(0).max(1).optional(),
  plumed: z
    .string()
    .min(1, "the plumed.dat snippet must not be empty")
    .max(4000, "the plumed.dat snippet exceeds 4000 characters")
    .optional(),
});

interface BridgeRequest {
  nFrames?: number;
  seed?: number;
  noise?: number;
  plumed?: string;
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
      payload: { op: "cv", ...(parsed.data as BridgeRequest) },
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "MD collective-variable analysis failed." },
      { status: 500 }
    );
  }
}

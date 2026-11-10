import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Round 257 -- the Molecular Dynamics tab's salt-bridge timeline card:
 * one POST runs the REAL pharmaml.md ionic-contact stack (getcontacts'
 * distance criterion) on the designed two-bridge demo --
 * synthetic_salt_bridge_trajectory builds two phase-shifted
 * cation-anion pairs (Lys NZ...Glu CD, Arg NH1...Asp CG) oscillating
 * across the 4.0 A cutoff, salt_bridge_timeline counts per frame --
 * every number the chart shows comes from the shipped toolkit, so the
 * web demo and the Python API cannot drift apart.
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
});

interface BridgeRequest {
  nFrames?: number;
  seed?: number;
  noise?: number;
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
      payload: { op: "saltbridge", ...(parsed.data as BridgeRequest) },
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "MD salt-bridge analysis failed." },
      { status: 500 }
    );
  }
}

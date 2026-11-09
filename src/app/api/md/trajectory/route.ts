import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Round 249 -- the Molecular Dynamics tab's trajectory lab: one POST
 * runs the REAL pharmaml.md analysis stack on the deterministic
 * synthetic alpha-helix (seeded noise + slow rotational drift):
 * rmsd_trajectory, radius_of_gyration_trajectory,
 * native_contact_fraction, rmsf, contact_map, trajectory_pca and
 * anm_fluctuations -- every number the chart shows comes from the
 * shipped toolkit, so the web demo and the Python API cannot drift
 * apart.
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
  contactCutoff: z.number().min(1).max(12).optional(),
});

interface BridgeRequest {
  nFrames?: number;
  seed?: number;
  noise?: number;
  contactCutoff?: number;
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
      payload: { op: "trajectory", ...(parsed.data as BridgeRequest) },
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "MD trajectory analysis failed." },
      { status: 500 }
    );
  }
}

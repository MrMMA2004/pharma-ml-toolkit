import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * MASTER_PROMPT part3 §24 -- Synthetic Accessibility scoring over the
 * Ertl-Schuffenhauer method, wired through the python bridge to
 * pharmaml.chem.sa_scoring (sa_scores -> filter_by_sa -> summary).
 * Lower score = estimated easier to synthesise; the payload always
 * carries the method, direction and limitations copy so the UI can
 * never present a bare number.
 */
const bodySchema = z
  .object({
    smiles: z
      .array(z.string().min(1, "SMILES strings must not be empty").max(500))
      .min(1, "pass at least one SMILES")
      .max(50, "the web calculator handles at most 50 molecules per run"),
    minSa: z.number().min(1).max(10).optional(),
    maxSa: z.number().min(1).max(10).optional(),
  })
  .refine((v) => v.minSa === undefined || v.maxSa === undefined || v.minSa <= v.maxSa, {
    message: "minSa must not exceed maxSa",
  });

interface BridgeRequest {
  smiles: string[];
  minSa?: number;
  maxSa?: number;
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
      script: "sa_score_bridge.py",
      payload: parsed.data as BridgeRequest,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "SA scoring failed." },
      { status: 500 }
    );
  }
}

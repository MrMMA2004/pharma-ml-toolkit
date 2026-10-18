import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Must match FEATURE_FAMILIES in pharmaml/chem/features.py. */
const FAMILY_VALUES = [
  "descriptors",
  "atompair_count",
  "ecfp",
  "avalon_count",
  "rdkit_fingerprint",
  "morgan_full",
  "morgan_scaffold",
] as const;

const bodySchema = z.object({
  smiles: z
    .array(z.string().min(1, "SMILES strings must not be empty").max(500))
    .min(1, "pass at least one SMILES")
    .max(20, "the web calculator handles at most 20 molecules per run"),
  families: z.array(z.enum(FAMILY_VALUES)).min(1).max(7).optional(),
  k: z.number().int().min(1).max(1500),
});

interface BridgeRequest {
  smiles: string[];
  families?: string[];
  k: number;
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
        detail: parsed.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`),
      },
      { status: 400 }
    );
  }

  try {
    const result = await runPythonBridge({
      script: "chem_features_bridge.py",
      payload: parsed.data as BridgeRequest,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Feature computation failed." },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const bodySchema = z.object({
  smiles: z
    .array(z.string().min(1, "SMILES strings must not be empty").max(500))
    .min(1, "pass at least one SMILES")
    .max(10, "the web calculator handles at most 10 ligands per run"),
  padding: z.number().min(0).max(20).default(4.0),
  spacing: z.number().min(0.1).max(2).default(0.375),
  gaRun: z.number().int().min(1).max(10000).default(100),
  gaPopSize: z.number().int().min(1).max(10000).default(150),
});

interface BridgeRequest {
  smiles: string[];
  padding: number;
  spacing: number;
  gaRun: number;
  gaPopSize: number;
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
      script: "docking_prep_bridge.py",
      payload: {
        smiles: parsed.data.smiles,
        padding: parsed.data.padding,
        spacing: parsed.data.spacing,
        gaRun: parsed.data.gaRun,
        gaPopSize: parsed.data.gaPopSize,
      } satisfies BridgeRequest,
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Ligand preparation failed." },
      { status: 500 }
    );
  }
}

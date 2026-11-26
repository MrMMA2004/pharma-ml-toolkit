import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Next.js web-console endpoint for two molecule-surgery demonstrations:
 * one-step ring-size mutants and seeded scaffold-hop crossover. Both call
 * the real pharmaml.generative API with RDKit on the server and return
 * structure grids. This endpoint is a separate interface surface, not a
 * feature-parity claim for the React/Tauri desktop app. Request caps match
 * scripts/surgery_bridge.py. */

const MAX_SMILES_CHARS = 300;
const MAX_CHILDREN = 8;
const MAX_SEED = 999_999;

const smilesField = z
  .string()
  .trim()
  .min(1, "the SMILES string must not be empty")
  .max(MAX_SMILES_CHARS, `the SMILES string must be at most ${MAX_SMILES_CHARS} characters`);

const bodySchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("topology"),
    smiles: smilesField,
  }),
  z.object({
    mode: z.literal("breed"),
    parentA: smilesField,
    parentB: smilesField,
    children: z
      .number()
      .int()
      .min(1, "children must be between 1 and 8")
      .max(MAX_CHILDREN, "children must be between 1 and 8"),
    seed: z
      .number()
      .int()
      .min(0)
      .max(MAX_SEED, `seed must be between 0 and ${MAX_SEED}`),
  }),
]);

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
      script: "surgery_bridge.py",
      payload: parsed.data,
      // the structure grid renders up to 24 RDKit panels to PNG
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Molecule surgery failed." },
      { status: 500 }
    );
  }
}

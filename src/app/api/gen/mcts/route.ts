import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** The web twin of the desktop Studio Generative page's MCTS method
 * ("monte carlo tree search (MCTS)" — the ChemTSv2-style
 * MctsOptimizer over an on-the-fly SMILES RNN prior): one POST runs
 * the REAL pharmaml.generative search with real RDKit + torch on the
 * server and returns the ranked leaderboard, the per-cycle history and
 * the top-molecule structure grid. Mirrors the bridge caps in
 * scripts/mcts_bridge.py. */

const MAX_ITERATIONS = 24;
const MIN_ITERATIONS = 4;
const MAX_SEED = 999_999;
const MAX_SMILES_CHARS = 300;
/** 200 corpus lines x 80 chars + newlines, with headroom — the bridge
 * truncates past 200 lines and says so in the summary. */
const MAX_CORPUS_CHARS = 20_000;

const bodySchema = z.object({
  objective: z.enum([
    "qed",
    "logp",
    "mw",
    "tpsa",
    "blend",
    "multi",
    "similarity",
    "sa",
  ]),
  iterations: z
    .number()
    .int()
    .min(
      MIN_ITERATIONS,
      `iterations must be between ${MIN_ITERATIONS} and ${MAX_ITERATIONS}`
    )
    .max(
      MAX_ITERATIONS,
      `iterations must be between ${MIN_ITERATIONS} and ${MAX_ITERATIONS}`
    ),
  treePolicy: z.enum(["uct", "puct"]).default("uct"),
  seed: z
    .number()
    .int()
    .min(0)
    .max(MAX_SEED, `seed must be between 0 and ${MAX_SEED}`),
  lo: z.number().finite().default(1.0),
  hi: z.number().finite().default(3.0),
  reference: z
    .string()
    .trim()
    .max(MAX_SMILES_CHARS, `reference must be at most ${MAX_SMILES_CHARS} characters`)
    .default(""),
  corpus: z.string().max(MAX_CORPUS_CHARS, "the corpus is too long").default(""),
  seeded: z.boolean().default(false),
});

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
      script: "mcts_bridge.py",
      payload: parsed.data,
      // iterations=24 with the 8-epoch RNN prior + the grid render
      // measures well under 30 s; 90 s matches the surgery route's
      // headroom for a cold first spawn.
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "MCTS search failed." },
      { status: 500 }
    );
  }
}

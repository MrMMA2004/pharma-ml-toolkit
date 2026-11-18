import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** The web twin of the desktop Studio Response AI page (the
 * `pharmaml.ml.drugresponse` module — CaDRReS-derived drug-response
 * learning): one POST runs the REAL module on the server through the
 * repo-root bridge. The card's four analyses map to the page's
 * headline tabs — hill (the 4PL fit), metrics (Pearson/Spearman/
 * RMSE/MAE + the Kendall trio), model (the seeded 12x4 screen CaDRReS
 * factorisation), cv (cell-line-grouped cross-validation). The page's
 * deeper diagnostics (informative regression, the response VAE,
 * multi-omics fusion) stay desktop-side. Mirrors the caps in
 * scripts/drugresponse_bridge.py. */

const MAX_SERIES_CHARS = 4_000;
const MIN_EPOCHS = 20;
const MAX_EPOCHS = 400;
const MAX_SEED = 999_999;

const bodySchema = z.object({
  analysis: z.enum(["hill", "metrics", "model", "cv"]),
  dose: z
    .string()
    .max(MAX_SERIES_CHARS, `dose must be at most ${MAX_SERIES_CHARS} characters`)
    .default(""),
  response: z
    .string()
    .max(MAX_SERIES_CHARS, `response must be at most ${MAX_SERIES_CHARS} characters`)
    .default(""),
  yTrue: z
    .string()
    .max(MAX_SERIES_CHARS, `yTrue must be at most ${MAX_SERIES_CHARS} characters`)
    .default(""),
  yPred: z
    .string()
    .max(MAX_SERIES_CHARS, `yPred must be at most ${MAX_SERIES_CHARS} characters`)
    .default(""),
  noise: z.number().finite().min(0).max(0.5).default(0.01),
  epochs: z
    .number()
    .int()
    .min(MIN_EPOCHS, `epochs must be between ${MIN_EPOCHS} and ${MAX_EPOCHS}`)
    .max(MAX_EPOCHS, `epochs must be between ${MIN_EPOCHS} and ${MAX_EPOCHS}`)
    .default(120),
  factors: z
    .number()
    .int()
    .min(1, "factors must be between 1 and 4")
    .max(4, "factors must be between 1 and 4")
    .default(2),
  folds: z
    .number()
    .int()
    .min(2, "folds must be between 2 and 8")
    .max(8, "folds must be between 2 and 8")
    .default(4),
  seed: z
    .number()
    .int()
    .min(0)
    .max(MAX_SEED, `seed must be between 0 and ${MAX_SEED}`)
    .default(42),
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
      script: "drugresponse_bridge.py",
      payload: parsed.data,
      // the heaviest leg (model, 400 epochs on the 12x4 screen)
      // measures well under 2 s; 60 s is generous headroom for a cold
      // first spawn of the toolkit venv.
      timeoutMs: 60_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const mapped = bridgeErrorResponse(err);
    if (mapped) {
      return NextResponse.json(mapped.body, { status: mapped.status });
    }
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "The drug-response analysis failed." },
      { status: 500 }
    );
  }
}

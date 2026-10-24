import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { bridgeErrorResponse, runPythonBridge } from "@/lib/python-bridge";
import type { SessionReportResponse } from "@/lib/session-report";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/* ------------------------------------------------------------------ */
/* Request schema — the section payload the client collector builds     */
/* (src/lib/session-report.ts).  The Python bridge validates the same  */
/* contract again; zod gives the client typed 400s before the spawn.   */
/* ------------------------------------------------------------------ */

const pointSchema = z.union([
  z.tuple([z.number(), z.number()]),
  z.tuple([z.number(), z.number(), z.number(), z.number()]),
]);

const traceSchema = z.object({
  name: z.string().min(1).max(200),
  style: z.enum(["line", "band", "scatter"]),
  points: z.array(pointSchema).min(1).max(2000),
  classes: z.array(z.number()).optional(),
  probas: z.array(z.number()).optional(),
});

const sectionSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("metrics"),
    title: z.string().min(1).max(200),
    items: z
      .array(z.tuple([z.string().min(1).max(200), z.string().max(200)]))
      .min(1)
      .max(40),
  }),
  z.object({
    kind: z.literal("figure"),
    title: z.string().min(1).max(200),
    xLabel: z.string().max(200),
    yLabel: z.string().max(200),
    traces: z.array(traceSchema).min(1).max(8),
  }),
  z.object({
    kind: z.literal("table"),
    title: z.string().min(1).max(200),
    columns: z.array(z.string().min(1).max(200)).min(1).max(12),
    rows: z
      .array(z.array(z.union([z.string().max(40), z.number()])).min(1).max(12))
      .min(1)
      .max(300),
  }),
]);

const bodySchema = z.object({
  theme: z.enum(["auto", "light", "dark"]),
  subtitle: z.string().max(200).default("web playground session"),
  sections: z.array(sectionSchema).min(1).max(16),
});

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = bodySchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "Invalid session-report request",
        detail: parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`),
      },
      { status: 400 },
    );
  }

  // The collector's metric values are already strings (fmtCell) — the
  // bridge pins the same contract.
  const payload = parsed.data;

  try {
    const result = await runPythonBridge<SessionReportResponse>({
      script: "report_bridge.py",
      payload,
      timeoutMs: 90_000,
    });
    return NextResponse.json(result);
  } catch (err) {
    const bridged = bridgeErrorResponse(err);
    if (bridged) return NextResponse.json(bridged.body, { status: bridged.status });
    return NextResponse.json(
      { error: "The report backend failed to run." },
      { status: 500 },
    );
  }
}

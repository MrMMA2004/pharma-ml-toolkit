import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { runPythonBridge } from "@/lib/python-bridge";
import type { DockingRescoreRow, DockingRescoreResponse, DockingRescoreVerdict } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const bodySchema = z.object({
  content: z.string().min(1, "content is required").max(200_000),
  columns: z
    .array(z.string().min(1))
    .min(1, "select at least one evidence column")
    .max(8, "at most 8 evidence columns"),
  weights: z
    .array(z.number().min(0).max(100))
    .max(8)
    .optional(),
  higher: z.array(z.string().min(1)).max(8).optional(),
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
        detail: parsed.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`),
      },
      { status: 400 }
    );
  }

  const { content, columns, weights, higher } = parsed.data;
  if (weights && weights.length !== columns.length) {
    return NextResponse.json(
      { error: `weights must match the ${columns.length} selected column(s).` },
      { status: 400 }
    );
  }

  try {
    const bridge = await runPythonBridge<Record<string, unknown>>({
      script: "docking_rescore_bridge.py",
      payload: { content, columns, weights, higher },
      timeoutMs: 30_000,
    });

    const rows = (bridge.rows as DockingRescoreRow[] | undefined) ?? [];
    const response: DockingRescoreResponse = {
      identityColumn: (bridge.identityColumn as string | undefined) ?? "",
      columns: (bridge.columns as string[] | undefined) ?? columns,
      rows,
      recipe: (bridge.recipe as string | undefined) ?? "",
      nRows: (bridge.nRows as number | undefined) ?? rows.length,
      verdict: (bridge.verdict as DockingRescoreVerdict | null | undefined) ?? null,
      pharmamlVersion: (bridge.pharmamlVersion as string | undefined) ?? "",
    };
    return NextResponse.json(response);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // Bridge validation failures (bad columns, degenerate data) are client
    // errors; infrastructure failures (missing script, timeout) are ours.
    if (message.startsWith("BRIDGE_ERROR:")) {
      return NextResponse.json(
        { error: message.slice("BRIDGE_ERROR:".length).trim() || "Rescoring failed." },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: message || "Could not rescore the supplied summary table." },
      { status: 500 }
    );
  }
}

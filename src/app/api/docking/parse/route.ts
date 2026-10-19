import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { parseDockingOutput } from "@/lib/docking-parse";
import type { DockingParseResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const bodySchema = z.object({
  format: z.enum(["vina", "dlg", "csv"]),
  content: z.string().min(1, "content is required").max(200_000),
  heavyAtoms: z.number().int().min(1).max(500).default(32),
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

  try {
    const result = parseDockingOutput(parsed.data.format, parsed.data.content, parsed.data.heavyAtoms);
    const response: DockingParseResponse = {
      format: result.format,
      poses: result.poses,
      stats: result.stats,
      chartData: result.chartData,
      consensus: result.consensus,
    };
    return NextResponse.json(response);
  } catch (err) {
    return NextResponse.json(
      {
        error:
          err instanceof Error
            ? err.message
            : "Could not parse the supplied docking output.",
      },
      { status: 400 }
    );
  }
}

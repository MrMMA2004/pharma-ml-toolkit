import { NextResponse } from "next/server";
import { measureZipBytes, readToolkitVersion } from "@/lib/toolkit-fs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * The byte size of the archive /api/toolkit/download streams — measured
 * with the SAME zip invocation and the SAME exclusion list (ZIP_EXCLUDE_ARGS
 * in toolkit-fs.ts), stdout counted in-process. Cached server-side for
 * 10 minutes with in-flight dedup, so repeated visitors get an instant
 * answer. Powers the honest "≈ N MB" badge on the download button —
 * previously the button displayed the UNCOMPRESSED tree size, which
 * overstated the real deflate-compressed download by ~2x.
 */
export async function GET() {
  try {
    const bytes = await measureZipBytes();
    if (bytes === null) {
      return NextResponse.json(
        { error: "Failed to measure the archive size." },
        { status: 500 }
      );
    }
    return NextResponse.json(
      { zipBytes: bytes, version: await readToolkitVersion() },
      { headers: { "Cache-Control": "no-store" } }
    );
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to measure the archive size.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

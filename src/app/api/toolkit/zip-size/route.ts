import { NextResponse } from "next/server";
import { measureZipBytes, openReleaseZip, readToolkitVersion } from "@/lib/toolkit-fs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * The byte size of the archive /api/toolkit/download streams right now.
 *
 * Release path (the normal case): the prebuilt release artifact is
 * opened + validated (openReleaseZip — the SAME handle-anchored gate the
 * download route applies, EOCD tail check included) and its fstat size
 * reported — the EXACT byte count the download sends, with zero spawns
 * and zero compression passes. A truncated artifact is refused here
 * exactly as the download refuses it, so the badge's exact/estimate flag
 * always matches the path that would serve. Flagged `exact: true` so the
 * download badge can drop its "≈" and the panel can switch to the stable-checksum wording.
 *
 * Fallback (release file missing): measured with the SAME zip invocation
 * and the SAME exclusion list the streamer uses (ZIP_EXCLUDE_ARGS in
 * toolkit-fs.ts), stdout counted in-process. Cached server-side for
 * 10 minutes with in-flight dedup, so repeated visitors get an instant
 * answer. Powers the honest "≈ N MB" badge on the download button —
 * previously the button displayed the UNCOMPRESSED tree size, which
 * overstated the real deflate-compressed download by ~2x.
 */
export async function GET() {
  try {
    const version = await readToolkitVersion();

    const release = await openReleaseZip(version);
    if (release) {
      const bytes = release.size;
      // No stream takes the handle on this path — close it here.
      await release.fh.close();
      return NextResponse.json(
        { zipBytes: bytes, version, exact: true, source: "release" },
        { headers: { "Cache-Control": "no-store" } }
      );
    }

    const bytes = await measureZipBytes();
    if (bytes === null) {
      return NextResponse.json(
        { error: "Failed to measure the archive size." },
        { status: 500 }
      );
    }
    return NextResponse.json(
      { zipBytes: bytes, version, exact: false, source: "stream" },
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

import { spawn } from "child_process";
import { createReadStream, promises as fs } from "fs";
import { Readable } from "stream";
import { NextResponse } from "next/server";
import {
  TOOLKIT_ROOT,
  ZIP_EXCLUDE_ARGS,
  readToolkitVersion,
  releaseZipAbsPath,
  statReleaseZip,
} from "@/lib/toolkit-fs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Stream the toolkit archive — the PREBUILT release artifact when it exists,
 * a fresh on-the-fly build otherwise.
 *
 * Release path (the normal case): the round-end builder leaves a
 * testzip-verified, leak-scanned archive at
 * download/pharma-ml-toolkit-v<version>.zip. It is streamed straight from
 * disk with Content-Length (the browser gets a real progress bar), a strong
 * ETag (repeat clicks revalidate — a matching If-None-Match is a zero-byte
 * 304), and the exact bytes of the shipped release: a stable SHA-256 per
 * version, so a checksum verified once matches every later download. This
 * replaced a rebuild-per-request streamer that once nested the historical
 * release zips (97 MB of already-compressed bytes) inside every download —
 * the gateway reset the connection mid-stream and the download died.
 *
 * Fallback (release file missing): `zip -r` over the tree piped to stdout
 * with the SAME exclusion list the release builder uses (ZIP_EXCLUDE_ARGS
 * in toolkit-fs.ts, including *.zip). This path regenerates per request —
 * no Content-Length, no stable hash, `no-store` — but it is fast (~1 s for
 * the ~13 MB archive) and stays correct while a release is mid-rebuild.
 *
 * X-Toolkit-Zip-Source tells tooling which path served ("release"|"stream")
 * and X-Toolkit-Version exposes the release without opening the archive.
 */
export async function GET(req: Request) {
  try {
    const info = await fs.stat(TOOLKIT_ROOT);
    if (!info.isDirectory()) {
      throw new Error("Toolkit directory is missing on the server.");
    }

    // Version the filename (and expose X-Toolkit-Version) so releases can be
    // told apart — and updated in place — without opening the archive.
    const version = await readToolkitVersion();

    // ---- Preferred: the prebuilt release artifact ---------------------
    const release = await statReleaseZip(version);
    if (release) {
      const etag = `"zip-v${version}-${release.bytes}-${Math.floor(release.mtimeMs)}"`;
      const headers: Record<string, string> = {
        ETag: etag,
        "X-Toolkit-Version": version,
        "X-Toolkit-Zip-Source": "release",
        // no-cache: the browser may store the body but must revalidate —
        // an in-place rebuild (new size/mtime) revalidates and re-streams.
        "Cache-Control": "no-cache",
      };
      if (req.headers.get("if-none-match") === etag) {
        return new Response(null, { status: 304, headers });
      }
      const nodeStream = createReadStream(releaseZipAbsPath(version));
      const webStream = Readable.toWeb(nodeStream) as unknown as ReadableStream<Uint8Array>;
      return new Response(webStream, {
        status: 200,
        headers: {
          ...headers,
          "Content-Type": "application/zip",
          "Content-Length": String(release.bytes),
          "Content-Disposition": `attachment; filename="pharma-ml-toolkit-v${version}.zip"`,
        },
      });
    }

    // ---- Fallback: on-the-fly build -----------------------------------
    const child = spawn(
      "zip",
      ["-r", "-q", "-", ".", "-x", ...ZIP_EXCLUDE_ARGS],
      { cwd: TOOLKIT_ROOT }
    );

    // Wait until the process actually spawned so launch failures (ENOENT
    // etc.) surface as a clean 500 before any bytes are sent.
    await new Promise<void>((resolve, reject) => {
      child.once("spawn", () => resolve());
      child.once("error", (err: NodeJS.ErrnoException) =>
        reject(new Error(`Failed to launch the zip utility: ${err.message}`))
      );
    });

    // Non-zero exit after streaming started can no longer change the
    // response — log it for diagnostics instead of throwing unhandled.
    let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString()).slice(-4096);
    });
    child.once("close", (code) => {
      if (code !== 0) {
        console.error(`[toolkit/download] zip exited with code ${code}: ${stderr.trim()}`);
      }
    });

    const webStream = Readable.toWeb(child.stdout) as unknown as ReadableStream<Uint8Array>;

    return new Response(webStream, {
      status: 200,
      headers: {
        "Content-Type": "application/zip",
        "Content-Disposition": `attachment; filename="pharma-ml-toolkit-v${version}.zip"`,
        "X-Toolkit-Version": version,
        "X-Toolkit-Zip-Source": "stream",
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to build the toolkit ZIP archive.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}
